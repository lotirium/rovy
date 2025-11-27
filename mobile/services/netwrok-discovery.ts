import * as Network from "expo-network";

export interface DiscoveredRobot {
  ip: string;
  port: number;
  baseUrl: string;
  name?: string;
  robotId?: string;
  wifiSsid?: string;
  responseTimeMs: number;
}

interface HealthResponse {
  status?: string;
  robot_id?: string;
  robotId?: string;
  name?: string;
  network?: {
    ssid?: string;
    wifiSsid?: string;
    ip?: string;
  };
  [key: string]: unknown;
}

const DEFAULT_PORT = 8000;
const PROBE_TIMEOUT_MS = 1500;
const MAX_CONCURRENT_PROBES = 20;

/**
 * Probes a single IP:port to check if a robot is running there.
 */
async function probeHost(
  ip: string,
  port: number = DEFAULT_PORT
): Promise<DiscoveredRobot | null> {
  const baseUrl = `http://${ip}:${port}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);

  const startTime = Date.now();

  try {
    const response = await fetch(`${baseUrl}/health`, {
      method: "GET",
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return null;
    }

    const responseTimeMs = Date.now() - startTime;

    let data: HealthResponse = {};
    try {
      data = await response.json();
    } catch {
      // Response OK but not JSON - still a valid robot
    }

    return {
      ip,
      port,
      baseUrl,
      name: data.name,
      robotId: data.robot_id || data.robotId,
      wifiSsid: data.network?.ssid || data.network?.wifiSsid,
      responseTimeMs,
    };
  } catch {
    clearTimeout(timeoutId);
    return null;
  }
}

/**
 * Extracts the subnet from an IP address (e.g., "192.168.1.50" -> "192.168.1")
 */
function getSubnet(ip: string): string | null {
  const parts = ip.split(".");
  if (parts.length !== 4) {
    return null;
  }
  return `${parts[0]}.${parts[1]}.${parts[2]}`;
}

/**
 * Generates IP addresses to scan on the local subnet.
 * Prioritizes common DHCP ranges and router-assigned IPs.
 */
function generateScanTargets(subnet: string): string[] {
  const targets: string[] = [];

  // Common static/reserved IPs for IoT devices
  const priorityHosts = [1, 2, 100, 101, 102, 123, 200, 201, 254];
  for (const host of priorityHosts) {
    targets.push(`${subnet}.${host}`);
  }

  // DHCP range (typically 100-200 or 2-254)
  for (let host = 2; host <= 254; host++) {
    const ip = `${subnet}.${host}`;
    if (!targets.includes(ip)) {
      targets.push(ip);
    }
  }

  return targets;
}

/**
 * Scans a batch of IPs concurrently.
 */
async function scanBatch(
  ips: string[],
  port: number
): Promise<DiscoveredRobot[]> {
  const results = await Promise.all(ips.map((ip) => probeHost(ip, port)));
  return results.filter((r): r is DiscoveredRobot => r !== null);
}

export interface DiscoveryOptions {
  /** Port to scan (default: 8000) */
  port?: number;
  /** Maximum IPs to scan (default: 254) */
  maxHosts?: number;
  /** Callback for progress updates */
  onProgress?: (
    scanned: number,
    total: number,
    found: DiscoveredRobot[]
  ) => void;
  /** Abort signal to cancel discovery */
  signal?: AbortSignal;
}

export interface DiscoveryResult {
  robots: DiscoveredRobot[];
  phoneIp: string | null;
  subnet: string | null;
  scannedCount: number;
  durationMs: number;
}

/**
 * Discovers robots on the local network by scanning the subnet.
 */
export async function discoverRobotsOnNetwork(
  options: DiscoveryOptions = {}
): Promise<DiscoveryResult> {
  const { port = DEFAULT_PORT, maxHosts = 254, onProgress, signal } = options;

  const startTime = Date.now();
  const foundRobots: DiscoveredRobot[] = [];

  // Get phone's IP address
  let phoneIp: string | null = null;
  try {
    phoneIp = await Network.getIpAddressAsync();
    if (phoneIp === "0.0.0.0" || !phoneIp) {
      phoneIp = null;
    }
  } catch (error) {
    console.warn("Failed to get phone IP:", error);
  }

  if (!phoneIp) {
    return {
      robots: [],
      phoneIp: null,
      subnet: null,
      scannedCount: 0,
      durationMs: Date.now() - startTime,
    };
  }

  const subnet = getSubnet(phoneIp);
  if (!subnet) {
    return {
      robots: [],
      phoneIp,
      subnet: null,
      scannedCount: 0,
      durationMs: Date.now() - startTime,
    };
  }

  const allTargets = generateScanTargets(subnet).slice(0, maxHosts);
  let scannedCount = 0;

  // Scan in batches
  for (let i = 0; i < allTargets.length; i += MAX_CONCURRENT_PROBES) {
    if (signal?.aborted) {
      break;
    }

    const batch = allTargets.slice(i, i + MAX_CONCURRENT_PROBES);
    const batchResults = await scanBatch(batch, port);

    foundRobots.push(...batchResults);
    scannedCount += batch.length;

    onProgress?.(scannedCount, allTargets.length, [...foundRobots]);

    // Early exit if we found robots (most users only have one)
    // Continue for a bit to find all robots
    if (foundRobots.length > 0 && scannedCount >= 50) {
      // Found at least one robot and scanned priority range
      break;
    }
  }

  return {
    robots: foundRobots,
    phoneIp,
    subnet,
    scannedCount,
    durationMs: Date.now() - startTime,
  };
}

/**
 * Quick discovery - only scans priority IPs for fast results.
 */
export async function quickDiscovery(
  options: Omit<DiscoveryOptions, "maxHosts"> = {}
): Promise<DiscoveryResult> {
  return discoverRobotsOnNetwork({
    ...options,
    maxHosts: 30, // Only scan first 30 priority IPs
  });
}

/**
 * Try to reach a robot at a specific hostname (e.g., "rovy.local").
 * mDNS resolution depends on the network and device support.
 */
export async function tryMdnsHostname(
  hostname: string = "rovy.local",
  port: number = DEFAULT_PORT
): Promise<DiscoveredRobot | null> {
  return probeHost(hostname, port);
}

/**
 * Known Tailscale IPs for ROVY robots.
 * Add your robot's Tailscale IP here for automatic discovery.
 */
const KNOWN_TAILSCALE_IPS = [
  "100.72.107.106", // ROVY Pi
];

/**
 * Try to reach robot via Tailscale IP.
 */
export async function tryTailscaleIps(
  port: number = DEFAULT_PORT
): Promise<DiscoveredRobot | null> {
  for (const ip of KNOWN_TAILSCALE_IPS) {
    const robot = await probeHost(ip, port);
    if (robot) {
      return robot;
    }
  }
  return null;
}

/**
 * Combined discovery: tries mDNS, Tailscale, then subnet scan.
 */
export async function discoverRobots(
  options: DiscoveryOptions = {}
): Promise<DiscoveryResult> {
  const startTime = Date.now();
  const foundRobots: DiscoveredRobot[] = [];
  const port = options.port || DEFAULT_PORT;

  // Get phone IP for context
  let phoneIp: string | null = null;
  try {
    phoneIp = await Network.getIpAddressAsync();
    if (phoneIp === "0.0.0.0") phoneIp = null;
  } catch {}

  // 1. Try mDNS first (fast if it works)
  console.log("Trying mDNS discovery (rovy.local)...");
  const mdnsRobot = await tryMdnsHostname("rovy.local", port);
  if (mdnsRobot) {
    console.log("Found robot via mDNS:", mdnsRobot.ip);
    foundRobots.push(mdnsRobot);
    options.onProgress?.(1, 1, foundRobots);
    return {
      robots: foundRobots,
      phoneIp,
      subnet: phoneIp ? getSubnet(phoneIp) : null,
      scannedCount: 1,
      durationMs: Date.now() - startTime,
    };
  }

  // 2. Try Tailscale IPs (works across networks)
  console.log("Trying Tailscale discovery...");
  const tailscaleRobot = await tryTailscaleIps(port);
  if (tailscaleRobot) {
    console.log("Found robot via Tailscale:", tailscaleRobot.ip);
    foundRobots.push(tailscaleRobot);
    options.onProgress?.(1, 1, foundRobots);
    return {
      robots: foundRobots,
      phoneIp,
      subnet: phoneIp ? getSubnet(phoneIp) : null,
      scannedCount: 1,
      durationMs: Date.now() - startTime,
    };
  }

  // 3. Fall back to subnet scan
  console.log("Falling back to subnet scan...");
  return discoverRobotsOnNetwork(options);
}
