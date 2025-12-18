#!/usr/bin/env python3
"""
Test script for Rovy dancing mode
Run this on the Raspberry Pi to test dance functionality
"""

import sys
import time
from robot.rover import Rover

def test_dance_styles():
    """Test all three dance styles."""
    print("=" * 60)
    print("🕺 ROVY DANCE MODE TEST")
    print("=" * 60)
    print("\nInitializing rover connection...")
    
    try:
        # Connect to rover (adjust port if needed)
        rover = Rover('/dev/ttyAMA0')
        time.sleep(2)
        
        print("✅ Rover connected!")
        print("\nTesting will run each dance style for 3 seconds")
        print("Make sure the robot has space to move!\n")
        
        input("Press ENTER to start tests (or Ctrl+C to abort)...")
        
        # Test 1: Party Dance
        print("\n" + "=" * 60)
        print("TEST 1: PARTY DANCE (3 seconds)")
        print("=" * 60)
        rover.display_lines([
            "TEST MODE",
            "Party Dance",
            "3 seconds",
            "🎉"
        ])
        time.sleep(1)
        rover.dance('party', 3)
        time.sleep(2)
        print("✅ Party dance complete!")
        
        # Test 2: Wiggle Dance
        print("\n" + "=" * 60)
        print("TEST 2: WIGGLE DANCE (3 seconds)")
        print("=" * 60)
        rover.display_lines([
            "TEST MODE",
            "Wiggle Dance",
            "3 seconds",
            "🐍"
        ])
        time.sleep(1)
        rover.dance('wiggle', 3)
        time.sleep(2)
        print("✅ Wiggle dance complete!")
        
        # Test 3: Spin Dance
        print("\n" + "=" * 60)
        print("TEST 3: SPIN DANCE (3 seconds)")
        print("=" * 60)
        rover.display_lines([
            "TEST MODE",
            "Spin Dance",
            "3 seconds",
            "🌀"
        ])
        time.sleep(1)
        rover.dance('spin', 3)
        time.sleep(2)
        print("✅ Spin dance complete!")
        
        # Final status
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nRover Status:")
        status = rover.get_status()
        print(f"  Battery: {status.get('voltage', 0):.2f}V")
        print(f"  Temperature: {status.get('temperature', 0):.1f}°C")
        
        # Cleanup
        rover.display_lines([
            "TEST COMPLETE",
            "All dances OK",
            f"Battery: {status.get('voltage', 0):.1f}V",
            "✅"
        ])
        time.sleep(2)
        rover.cleanup()
        
        print("\n✅ Test complete! Robot stopped and cleaned up.")
        print("\nNext steps:")
        print("  1. Test via REST API: curl -X POST http://localhost:8000/dance -d '{\"style\":\"party\"}'")
        print("  2. Test via voice: Say 'Hey Rovy, dance!'")
        print("  3. Test from phone app")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        rover.stop()
        rover.cleanup()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if 'rover' in locals():
            rover.stop()
            rover.cleanup()
        
        sys.exit(1)


def test_rest_api():
    """Test the REST API endpoint."""
    print("\n" + "=" * 60)
    print("🌐 REST API TEST")
    print("=" * 60)
    
    try:
        import requests
        
        print("\nTesting /dance endpoint...")
        
        # Test with default parameters
        response = requests.post(
            'http://localhost:8000/dance',
            json={'style': 'party', 'duration': 3}
        )
        
        if response.status_code == 200:
            print("✅ REST API test passed!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ REST API returned status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except ImportError:
        print("⚠️  requests module not installed, skipping REST API test")
        print("   Install with: pip install requests")
    except Exception as e:
        print(f"❌ REST API test failed: {e}")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              ROVY DANCE MODE TEST SUITE                   ║
    ║                                                            ║
    ║  This will test all three dance styles:                   ║
    ║    1. Party Dance  - Spins, lights, wiggling             ║
    ║    2. Wiggle Dance - Side-to-side with head moves        ║
    ║    3. Spin Dance   - 360° spins with lights              ║
    ║                                                            ║
    ║  Each test runs for 3 seconds.                            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        test_dance_styles()
        
        # Optionally test REST API if requests is available
        try:
            import requests
            print("\n" + "=" * 60)
            response = input("\nTest REST API endpoint? (y/N): ").strip().lower()
            if response == 'y':
                test_rest_api()
        except ImportError:
            pass
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

