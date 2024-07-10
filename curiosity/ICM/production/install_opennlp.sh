#!/bin/bash

# Update package list and upgrade all packages
sudo apt update && sudo apt upgrade -y

# Install OpenJDK 17
sudo apt install -y openjdk-17-jdk

# Check if Java is installed
if type -p java; then
    echo "Java is installed."
    _java=java
elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]]; then
    echo "Java is installed."
    _java="$JAVA_HOME/bin/java"
else
    echo "Java is not installed. Please install Java 17 or higher."
    exit 1
fi

# Check Java version
if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    echo "Java version: $version"
    if [[ "$version" < "17" ]]; then
        echo "Java version is less than 17. Please upgrade Java."
        exit 1
    fi
fi

# Install Maven
sudo apt install -y maven

# Check if Maven is installed
if type -p mvn; then
    echo "Maven is installed."
else
    echo "Maven is not installed. Please install Maven 3.3.9 or higher."
    exit 1
fi

# Install Git
sudo apt install -y git

# Clone the OpenNLP repository
git clone https://github.com/apache/opennlp.git

# Navigate to the OpenNLP directory
cd opennlp

# Build OpenNLP using Maven
mvn clean install -Dmaven.test.skip=true

# Navigate back to the parent directory
cd ..

# Suggested file name for the script
echo "Script completed. Save this script as 'install_opennlp.sh'."
