#!/bin/bash

# Get the current connected network SSID
NETWORK_SSID=$(nmcli -t -f active,ssid dev wifi | grep '^yes' | cut -d':' -f2)

# Replace these with your desired IP settings
DESIRED_IP="192.168.0.112/24" # Example: 192.168.0.111/24
GATEWAY_IP="192.168.0.1" # Example: 192.168.0.1
DNS_SERVER="8.8.8.8" # Or your preferred DNS server

if [ -z "$NETWORK_SSID" ]; then
    echo "No active Wi-Fi connection found."
    exit 1
fi

echo "Modifying network settings for SSID: $NETWORK_SSID"

# Modify the network connection to use the static IP address
sudo nmcli con mod "$NETWORK_SSID" ipv4.addresses "$DESIRED_IP"

# Set the gateway address
sudo nmcli con mod "$NETWORK_SSID" ipv4.gateway "$GATEWAY_IP"

# Set the DNS server
sudo nmcli con mod "$NETWORK_SSID" ipv4.dns "$DNS_SERVER"

# Change the method to manual to enable static IP
sudo nmcli con mod "$NETWORK_SSID" ipv4.method manual

# Apply the changes by bringing the connection up
sudo nmcli con up "$NETWORK_SSID"
