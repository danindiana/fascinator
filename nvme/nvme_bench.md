https://ssd.userbenchmark.com/Samsung-990-Pro-M2-2TB/Rating/4161

To benchmark the SAMSUNG SSD 990 PRO 2TB, PCIe 4.0 M.2 2280 with sequential read speeds up to 7,450MB/s on an Ubuntu 22.04 server, we can create a C++ program that measures the time taken to read a large file from the SSD. This program will perform the following steps:

1. **Preparation**: Generate a large test file on the SSD to be used for the benchmark. The size of this file should be significant enough (e.g., several GBs) to accurately measure read speeds and account for any caching effects.

2. **Benchmarking Process**:
   - Open the test file using C++ file I/O operations.
   - Read the file in chunks (e.g., 64MB chunks) to measure the time taken for each read operation.
   - Calculate the average read speed based on the total time taken and the size of the test file.

3. **Cleanup**: Optionally, delete the test file after the benchmarking is complete to free up space.

4. **Reporting**: Output the benchmark results, including the average read speed, to the console or a log file.

### C++ Benchmarking Program Pseudocode

```cpp
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

const size_t CHUNK_SIZE = 64 * 1024 * 1024; // 64MB
const std::string TEST_FILE_PATH = "/path/to/test/file";

void generateTestFile(const std::string& filePath, size_t fileSize) {
    // Function to generate a large test file of specified size
}

double benchmarkReadSpeed(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading.\n";
        return 0.0;
    }

    std::vector<char> buffer(CHUNK_SIZE);
    auto start = std::chrono::high_resolution_clock::now();

    size_t totalBytesRead = 0;
    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
        totalBytesRead += file.gcount();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    file.close();

    return totalBytesRead / duration.count() / (1024 * 1024); // MB/s
}

int main() {
    // Optionally, generate a test file
    // generateTestFile(TEST_FILE_PATH, desiredFileSize);

    double readSpeed = benchmarkReadSpeed(TEST_FILE_PATH);
    std::cout << "Average Read Speed: " << readSpeed << " MB/s\n";

    // Cleanup test file if necessary
    // remove(TEST_FILE_PATH.c_str());

    return 0;
}
```

### Notes:
- **File Generation**: The `generateTestFile` function is a placeholder. In practice, you might use `dd` or another tool to create a large file filled with random data.
- **File Path**: Replace `/path/to/test/file` with the actual path where the test file is located on the SSD.
- **Chunk Size**: The chunk size (64MB in this example) can be adjusted based on the system's memory availability and to fine-tune the benchmark.
- **Compilation**: Compile the program with optimizations enabled (e.g., `g++ -O3 -o ssd_benchmark ssd_benchmark.cpp`).
- **Running the Benchmark**: Ensure the test file is located on the SSD you wish to benchmark. Also, consider running the benchmark multiple times to account for variations in system load and SSD thermal conditions.

To optimize the C++ benchmarking program for measuring the read speed of your SAMSUNG SSD 990 PRO 2TB on an Ubuntu 22.04 server, consider the following enhancements:

1. **Direct I/O**: Use direct I/O to bypass the operating system's cache, ensuring that your benchmark measures the SSD's performance rather than the efficiency of the OS cache. This can be achieved by opening the file with the `O_DIRECT` flag.

2. **Threaded Reading**: Utilize multiple threads to perform read operations in parallel, simulating real-world high-concurrency workloads and taking full advantage of SSD's capabilities.

3. **Large File Size**: Ensure the test file is significantly larger than the system's RAM to prevent caching effects. A file size of at least twice the amount of physical RAM is recommended.

4. **Aligned Memory and I/O**: For direct I/O, it's crucial that memory buffers are aligned to the filesystem's block size. This alignment can improve I/O efficiency.

5. **Warm-up Phase**: Include a warm-up phase before the actual benchmarking to stabilize the performance.

6. **Measure Latency**: In addition to throughput, measure the latency of each read operation to get a more comprehensive view of the SSD's performance.

7. **Avoid System Interference**: Run the benchmark in a "quiet" system state with minimal other activities to reduce interference.

8. **Repeat Measurements**: Perform multiple runs and calculate the average to account for variability in measurements.

### Optimized C++ Benchmarking Program Example

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

const size_t CHUNK_SIZE = 64 * 1024 * 1024; // 64MB
const int NUM_THREADS = 4; // Adjust based on your system's capabilities
const std::string TEST_FILE_PATH = "/path/to/test/file";

// Thread function for performing parallel reads
void readThread(int fd, size_t fileSize, std::atomic<size_t>& totalBytesRead) {
    // Allocate aligned memory for direct I/O
    void* buf;
    posix_memalign(&buf, 4096, CHUNK_SIZE); // Align to 4096 bytes (typical page/block size)

    size_t bytesRead = 0;
    while (bytesRead < fileSize) {
        ssize_t result = read(fd, buf, CHUNK_SIZE);
        if (result <= 0) break; // End of file or error
        bytesRead += result;
        totalBytesRead += result;
    }

    free(buf);
}

double benchmarkReadSpeed(const std::string& filePath, size_t fileSize) {
    int fd = open(filePath.c_str(), O_RDONLY | O_DIRECT); // Open with O_DIRECT for direct I/O
    if (fd < 0) {
        std::cerr << "Failed to open file for reading.\n";
        return 0.0;
    }

    std::atomic<size_t> totalBytesRead(0);
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    // Create and start threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(readThread, fd, fileSize / NUM_THREADS, std::ref(totalBytesRead));
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    close(fd);

    return totalBytesRead / duration.count() / (1024 * 1024); // MB/s
}

int main() {
    // Ensure the test file exists and is large enough
    double readSpeed = benchmarkReadSpeed(TEST_FILE_PATH, fileSize);
    std::cout << "Average Read Speed: " << readSpeed << " MB/s\n";

    return 0;
}
```

### Compilation and Execution
- **Compilation**: Compile the program with optimizations and thread support, e.g., `g++ -O3 -pthread -o ssd_benchmark_optimized ssd_benchmark_optimized.cpp`.
- **Execution**: Run the program, ensuring that the test file path and size are correctly set.

### Final Notes
- **Direct I/O Requirements**: The file and buffer sizes must be aligned to the block size of the filesystem. This example assumes a 4096-byte block size, which is common, but you should verify this for your system.
- **System State**: For the most accurate results, ensure other system activities are minimized during benchmarking.
- **Permissions**: Running direct I/O operations might require elevated permissions or specific user group memberships.

How can I enable TRIM support for my SAMSUNG SSD 990 PRO 2TB on Ubuntu 22.04?

To enable TRIM support for your SAMSUNG SSD 990 PRO 2TB on Ubuntu 22.04, you can use the `fstrim` utility. TRIM is an ATA command that allows the operating system to inform an SSD which blocks of data are no longer considered in use and can be wiped internally. Enabling TRIM can help maintain the performance of your SSD over time. There are two main ways to enable TRIM on Ubuntu: manually running the `fstrim` command or setting up a scheduled job with `cron` or `systemd`.

### Manually Running TRIM with `fstrim`

1. **Open a Terminal**: You can do this by searching for "Terminal" in your applications menu or by pressing `Ctrl+Alt+T`.

2. **Run `fstrim` Manually**: To manually run TRIM on all mounted filesystems that support it, use the following command:
   ```bash
   sudo fstrim -a
   ```
   To run TRIM on a specific SSD, specify its mount point, for example:
   ```bash
   sudo fstrim / -v
   ```
   The `-v` option makes the command verbose, showing the amount of space that was trimmed.

### Automatically Running TRIM

Ubuntu uses `systemd`'s `fstrim.timer` to automatically trigger TRIM on a weekly basis. This should be enabled by default on Ubuntu 22.04, but you can verify and enable it manually if needed.

1. **Check if `fstrim.timer` is Active**: Run the following command to see if the timer is active:
   ```bash
   systemctl status fstrim.timer
   ```
   
2. **Enable `fstrim.timer` if Necessary**: If the timer is not already active, enable it with the following command:
   ```bash
   sudo systemctl enable fstrim.timer
   ```
   
3. **Start `fstrim.timer`**: To start the timer immediately without rebooting, use:
   ```bash
   sudo systemctl start fstrim.timer
   ```

### Verifying TRIM Support

To verify that TRIM is enabled and working, you can use the `lsblk` command with the `-D` option to check the `DISC-GRAN` (discard granularity) and `DISC-MAX` (discard max bytes) values, which should be non-zero for drives that have TRIM enabled.

```bash
lsblk -D
```

Look for your SSD in the output. Non-zero values in the `DISC-GRAN` and `DISC-MAX` columns indicate that TRIM is supported and enabled.

### Note

- **Data Safety**: While TRIM is generally safe, it's always a good idea to have up-to-date backups of your important data.
- **Performance**: Running TRIM, especially manually and frequently, is not typically necessary with modern SSDs and filesystems, as automatic TRIM and the SSD's own garbage collection processes are quite effective.
