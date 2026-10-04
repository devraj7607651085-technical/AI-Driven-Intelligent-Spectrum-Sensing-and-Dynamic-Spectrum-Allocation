import random

# Number of available channels
NUM_CHANNELS = 10

# Generate channel status
# 0 = FREE
# 1 = BUSY
random.seed(42)

channels = {}

for i in range(1, NUM_CHANNELS + 1):
    channels[i] = random.randint(0, 1)

print("\n==========================================")
print("DYNAMIC SPECTRUM ALLOCATION")
print("==========================================")

print("\nChannel Status:")
print("------------------------------------------")

for channel, status in channels.items():

    if status == 0:
        print(f"Channel {channel:2d} : FREE")
    else:
        print(f"Channel {channel:2d} : BUSY")

# Find free channels
free_channels = [
    channel for channel, status in channels.items()
    if status == 0
]

print("\n------------------------------------------")
print("Available Free Channels:")
print("------------------------------------------")

if free_channels:
    print(free_channels)

    # Allocate first available free channel
    selected_channel = free_channels[0]

    print("\n------------------------------------------")
    print("CHANNEL ALLOCATION")
    print("------------------------------------------")
    print(f"Selected Channel : Channel {selected_channel}")
    print("Status           : ALLOCATED")
    print("Reason           : Channel is FREE")

else:
    print("No free channel available.")
    print("Allocation Status : FAILED")

print("\n==========================================")