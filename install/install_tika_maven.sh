#!/bin/bash

# Update package list and upgrade all packages to their latest versions
sudo apt update && sudo apt upgrade -y

# Install OpenJDK 11 (required for Apache Maven and Apache Tika)
sudo apt install -y openjdk-11-jdk

# Install Apache Maven
sudo apt install -y maven

# Verify Maven installation
mvn -version

# Download Apache Tika server
wget https://dlcdn.apache.org/tika/2.7.0/tika-server-standard-2.7.0.jar -O tika-server.jar

# Move Tika server jar to a suitable location (e.g., /opt/tika)
sudo mkdir -p /opt/tika
sudo mv tika-server.jar /opt/tika/

# Create a systemd service for Tika server
sudo tee /etc/systemd/system/tika.service <<EOF
[Unit]
Description=Apache Tika Server
After=network.target

[Service]
User=root
ExecStart=/usr/bin/java -jar /opt/tika/tika-server.jar
SuccessExitStatus=143
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable and start the Tika service
sudo systemctl enable tika.service
sudo systemctl start tika.service

# Check the status of the Tika service
sudo systemctl status tika.service

echo "Apache Maven and Apache Tika installation and setup complete."
