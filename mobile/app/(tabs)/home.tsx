import { Image } from "expo-image";
import { useRouter } from "expo-router";
import React from "react";
import { Pressable, ScrollView, StyleSheet, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { ThemedText } from "@/components/themed-text";
import { ThemedView } from "@/components/themed-view";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { useRobot } from "@/context/robot-provider";

const PRIMARY_ACTIONS = [
  {
    id: "stream",
    label: "Live View",
    description: "See through my eyes",
    href: "/manual" as const,
    icon: "video.fill" as const,
    iconBg: "#3B82F6",
    iconColor: "#FFFFFF",
  },
  {
    id: "patrol",
    label: "Patrol",
    description: "Guard & explore",
    href: "/manual" as const,
    icon: "shield.fill" as const,
    iconBg: "#F59E0B",
    iconColor: "#FFFFFF",
  },
  {
    id: "follow",
    label: "Follow Me",
    description: "Stay by your side",
    href: "/manual" as const,
    icon: "figure.walk" as const,
    iconBg: "#8B5CF6",
    iconColor: "#FFFFFF",
  },
  {
    id: "detect",
    label: "Detect",
    description: "Find people & objects",
    href: "/manual" as const,
    icon: "person.crop.rectangle" as const,
    iconBg: "#EC4899",
    iconColor: "#FFFFFF",
  },
  {
    id: "goto",
    label: "Go To",
    description: "Navigate somewhere",
    href: "/manual" as const,
    icon: "location.fill" as const,
    iconBg: "#10B981",
    iconColor: "#FFFFFF",
  },
  {
    id: "snapshot",
    label: "Snapshot",
    description: "Capture a photo",
    href: "/manual" as const,
    icon: "camera.fill" as const,
    iconBg: "#EF4444",
    iconColor: "#FFFFFF",
  },
] as const;

const SECONDARY_ACTIONS = [
  {
    id: "drive",
    label: "Drive",
    icon: "arrow.up.arrow.down" as const,
    href: "/manual" as const,
  },
  {
    id: "memory",
    label: "Memory",
    icon: "brain" as const,
    href: "/(tabs)/status" as const,
  },
  {
    id: "settings",
    label: "Settings",
    icon: "gearshape.fill" as const,
    href: "/(tabs)/settings" as const,
  },
] as const;

export default function HomeScreen() {
  const { status } = useRobot();
  const router = useRouter();

  const batteryRaw =
    status?.battery ?? status?.telemetry?.battery ?? status?.health?.battery;
  const batteryLevel =
    typeof batteryRaw === "number" ? Math.round(batteryRaw) : undefined;
  const batteryLabel = batteryLevel !== undefined ? `${batteryLevel}%` : "—";
  const batteryColor =
    batteryLevel === undefined
      ? "#67686C"
      : batteryLevel >= 60
        ? "#34D399"
        : batteryLevel >= 30
          ? "#FBBF24"
          : "#EF4444";

  const isOnline = Boolean(status?.network?.ip);
  const wifiLabel =
    status?.network?.wifiSsid ??
    status?.network?.ssid ??
    (isOnline ? "Connected" : "Offline");

  return (
    <SafeAreaView style={styles.safeArea} edges={["top"]}>
      <ThemedView style={styles.screen}>
        <ScrollView
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
        >
          {/* Header with robot identity */}
          <View style={styles.header}>
            <View style={styles.robotIdentity}>
              <View style={styles.avatarContainer}>
                <Image
                  source={require("@/assets/images/rovy.png")}
                  style={styles.avatar}
                  contentFit="cover"
                />
                <View
                  style={[
                    styles.statusDot,
                    { backgroundColor: isOnline ? "#34D399" : "#EF4444" },
                  ]}
                />
              </View>
              <View style={styles.robotInfo}>
                <ThemedText style={styles.robotName}>JARVIS</ThemedText>
                <ThemedText style={styles.robotSubtitle}>
                  Your AI Robot Assistant
                </ThemedText>
              </View>
              <View style={styles.statusChip}>
                <IconSymbol name="battery.75" color={batteryColor} size={14} />
                <ThemedText style={styles.statusChipText}>{batteryLabel}</ThemedText>
              </View>
            </View>
            <View style={styles.statusDivider} />
            <View style={styles.statusItem}>
              <IconSymbol
                name="wifi"
                color={isOnline ? "#34D399" : "#67686C"}
                size={16}
              />
              <ThemedText style={styles.statusText}>{wifiLabel}</ThemedText>
            </View>
            {status?.network?.ip && (
              <>
                <View style={styles.statusDivider} />
                <ThemedText style={styles.statusIp}>
                  {status.network.ip}
                </ThemedText>
              </>
            )}
          </View>

          {/* Main Talk button - primary CTA */}
          <Pressable
            style={({ pressed }) => [
              styles.talkButton,
              pressed && styles.talkButtonPressed,
            ]}
            onPress={() => router.push("/agentic")}
          >
            <View style={styles.talkIconContainer}>
              <IconSymbol name="mic.fill" size={32} color="#04110B" />
            </View>
            <View style={styles.talkContent}>
              <ThemedText style={styles.talkLabel}>Talk to JARVIS</ThemedText>
              <ThemedText style={styles.talkHint}>
                Tap to start a voice conversation
              </ThemedText>
            </View>
            <IconSymbol name="chevron.right" size={20} color="#04110B" />
          </Pressable>

          {/* Feature grid */}
          <View style={styles.sectionHeader}>
            <ThemedText style={styles.sectionTitle}>Capabilities</ThemedText>
          </View>

          <View style={styles.featureGrid}>
            {ROBOT_FEATURES.map((feature) => (
              <Pressable
                key={feature.id}
                style={({ pressed }) => [
                  styles.featureCard,
                  pressed && styles.featureCardPressed,
                ]}
                onPress={() => router.push(feature.href)}
              >
                <View
                  style={[
                    styles.featureIcon,
                    { backgroundColor: feature.iconBg },
                  ]}
                >
                  <IconSymbol
                    name={feature.icon}
                    size={20}
                    color={feature.iconColor}
                  />
                </View>
                <ThemedText style={styles.featureLabel}>
                  {feature.label}
                </ThemedText>
                <ThemedText style={styles.featureDescription}>
                  {feature.description}
                </ThemedText>
              </Pressable>
            ))}
          </Animated.View>

          {/* Secondary Actions - Compact */}
          <Animated.View
            entering={FadeInDown.delay(500).duration(500)}
            style={styles.secondarySection}
          >
            {SECONDARY_ACTIONS.map((action, index) => (
              <Animated.View
                key={action.id}
                style={({ pressed }) => [
                  styles.quickAction,
                  pressed && styles.quickActionPressed,
                ]}
                onPress={() => router.push(action.href)}
              >
                <IconSymbol name={action.icon} size={20} color="#D1D5DB" />
                <ThemedText style={styles.quickActionLabel}>
                  {action.label}
                </ThemedText>
              </Pressable>
            ))}
          </View>
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#0F0F0F",
  },
  screen: {
    flex: 1,
    backgroundColor: "#0F0F0F",
  },
  content: {
    padding: 24,
    paddingTop: 8,
    paddingBottom: 48,
    gap: 24,
  },

  // Status Section
  statusSection: {
    gap: 20,
  },
  header: {
    marginBottom: 4,
  },
  robotIdentity: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
  },
  avatarContainer: {
    position: "relative",
  },
  avatar: {
    width: 64,
    height: 64,
    borderWidth: 2,
    borderColor: "#1DD1A1",
  },
  statusDot: {
    position: "absolute",
    bottom: 2,
    right: 2,
    width: 14,
    height: 14,
    borderWidth: 2,
    borderColor: "#0F0F0F",
  },
  robotInfo: {
    flex: 1,
  },
  robotName: {
    fontSize: 28,
    fontFamily: "JetBrainsMono_700Bold",
    color: "#F9FAFB",
    letterSpacing: -0.5,
  },
  robotSubtitle: {
    fontSize: 14,
    color: "#67686C",
    marginTop: 2,
  },
  statusBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#1A1A1A",
    paddingVertical: 12,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: 'rgba(26, 26, 26, 0.6)',
    borderWidth: 1,
    borderColor: "#252525",
  },
  statusItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  statusChipText: {
    fontSize: 13,
    color: "#D1D5DB",
    fontFamily: "JetBrainsMono_500Medium",
  },
  statusIp: {
    fontSize: 12,
    color: "#67686C",
    fontFamily: "JetBrainsMono_400Regular",
  },
  statusDivider: {
    width: 1,
    height: 16,
    backgroundColor: "#303030",
    marginHorizontal: 12,
  },
  talkButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#1DD1A1",
    padding: 20,
    gap: 16,
  },
  talkButtonPressed: {
    backgroundColor: "#17B891",
  },
  talkIconContainer: {
    width: 56,
    height: 56,
    backgroundColor: "rgba(4, 17, 11, 0.15)",
    alignItems: "center",
    justifyContent: "center",
  },
  primaryTextContainer: {
    flex: 1,
    gap: 4,
  },
  primaryLabel: {
    fontSize: 18,
    fontFamily: "JetBrainsMono_700Bold",
    color: "#04110B",
  },
  talkHint: {
    fontSize: 13,
    color: "#04110B",
    opacity: 0.7,
    marginTop: 2,
  },
  sectionHeader: {
    marginTop: 8,
  },
  sectionTitle: {
    fontSize: 13,
    fontFamily: "JetBrainsMono_600SemiBold",
    color: "#67686C",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  featureGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  featureCard: {
    width: "31.5%",
    backgroundColor: "#1A1A1A",
    padding: 14,
    borderWidth: 1,
    borderColor: "#252525",
    gap: 8,
  },
  featureCardPressed: {
    backgroundColor: "#222222",
    borderColor: "#353535",
  },
  featureIcon: {
    width: 40,
    height: 40,

    alignItems: "center",
    justifyContent: "center",
  },
  featureLabel: {
    fontSize: 14,
    fontFamily: "JetBrainsMono_600SemiBold",
    color: "#F9FAFB",
  },
  featureDescription: {
    fontSize: 11,
    color: "#67686C",
    lineHeight: 14,
  },
  quickActionsRow: {
    flexDirection: "row",
    gap: 12,
  },
  quickAction: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: "#1A1A1A",
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: "#252525",
  },
  quickActionPressed: {
    backgroundColor: "#222222",
    borderColor: "#353535",
  },
  secondaryLabel: {
    fontSize: 14,
    fontFamily: "JetBrainsMono_500Medium",
    color: "#D1D5DB",
  },
});
