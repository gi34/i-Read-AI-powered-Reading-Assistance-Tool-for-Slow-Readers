import time
import psutil
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Set a non-GUI backend to avoid issues with the main thread
import matplotlib
matplotlib.use('Agg')  # This sets the 'Agg' backend, which doesn't require a GUI

path = "stress_log.txt"

class StressLogger:
    def __init__(self, log_file=path):
        self.log_file = log_file
        self.process = psutil.Process()
        self.start_times = {}
        self.stage_durations = []
        self._write_header()

    def _write_header(self):
        with open(self.log_file, 'w') as f:
            f.write("STRESS TEST LOG\n=====================\n\n")

    def mark(self, stage_name):
        self.start_times[stage_name] = time.time()
        self._log(f"[{stage_name}] Start at {self.start_times[stage_name]:.4f}s")

    def measure(self, stage_name):
        if stage_name not in self.start_times:
            self._log(f"[{stage_name}] No start time recorded!")
            return
        elapsed = time.time() - self.start_times[stage_name]
        self.stage_durations.append((stage_name, elapsed, datetime.now()))  
        self._log(f"[{stage_name}] Elapsed time: {elapsed:.4f}s")

    def log_cpu_memory(self):
        cpu = self.process.cpu_percent(interval=0.1)
        mem = self.process.memory_info().rss / (1024 * 1024)  # MB
        self._log(f"[Resource] CPU: {cpu:.2f}% | Memory: {mem:.2f} MB")

    def request_highlight(self, word_index, target_word):
        stage_name = f"FullHighlight"
        
        # Mark the start time for this stage
        self.mark(stage_name)
        self._log(f"Highlight requested for: {target_word}")
        
        # Measure the duration at this point if needed (after the request)
        self.measure(stage_name)

    def send_highlight_event(self, chunk_index, word_index, target_word):
        key = f"FullHighlight"
        sent_time = time.time()
        
        # Call mark to record start time first
        self.mark(key)
        
        # Log the highlight event with the target word and chunk information
        self._log(f"[{key}] Highlighted word: {target_word} at chunk {chunk_index}, word index {word_index}, time {sent_time}")
        
        # Now measure the duration after the event has been recorded
        self.measure(key)
        
        # Log resource usage (CPU and Memory)
        self.log_cpu_memory()




    def plot_resource_usage(self):
        """This method is called to plot the resource usage."""
        self._plot_cpu_mem_graph()
        self._plot_stage_timings()

    def _plot_cpu_mem_graph(self):
        """Plot CPU and Memory usage over time."""
        timestamps, cpu_values, mem_values = [], [], []
        pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] \[Resource\] CPU: ([\d.]+)% \| Memory: ([\d.]+) MB")

        with open(self.log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    time_str, cpu, mem = match.groups()
                    timestamps.append(datetime.strptime(time_str, "%H:%M:%S"))
                    cpu_values.append(float(cpu))
                    mem_values.append(float(mem))

        if not timestamps:
            #print("No CPU/memory data to plot.")
            return

        # Direct plotting without threading, ensure to do this in the main thread
        self._generate_cpu_mem_plot(timestamps, cpu_values, mem_values)

    def _generate_cpu_mem_plot(self, timestamps, cpu_values, mem_values):
        """Generate and save the CPU and memory plot."""
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, cpu_values, marker='o', label="CPU (%)", color="tab:blue")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU and Memory Usage Over Time")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(timestamps, mem_values, marker='x', label="Memory (MB)", color="tab:green")
        plt.xlabel("Time")
        plt.ylabel("Memory Usage (MB)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("cpu_and_memory_usage.png")  # ✅ Saves the graph instead of showing it
        plt.close()

    def _plot_stage_timings(self):
        """Plot stage timings."""
        if not self.stage_durations:
            #print("No stage timing data to plot.")
            return

        # Group data by stage name
        stages = {}
        for stage_name, duration, timestamp in self.stage_durations:
            stages.setdefault(stage_name, []).append((timestamp, duration))

        plt.figure(figsize=(12, 6))
        for stage_name, entries in stages.items():
            times = [t for t, _ in entries]
            durations = [d for _, d in entries]
            plt.plot(times, durations, marker='o', label=stage_name)

        plt.title("Stage Timing Durations")
        plt.xlabel("Time")
        plt.ylabel("Elapsed Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("stage_timing.png")  # ✅ Saves the graph instead of showing it
        plt.close()

    def _log(self, message):
        """Log a message with a timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        #print(log_entry.strip())
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
