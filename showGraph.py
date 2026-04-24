import matplotlib.pyplot as plt

# Read the data
words = []
latencies = []

with open("highlight_latency_log.txt", "r") as file:
    for line in file:
        try:
            word, latency_str = line.strip().split(": ")
            latency = float(latency_str.replace(" s", ""))
            words.append(word)
            latencies.append(latency)
        except ValueError:
            print(f"Skipping line: {line.strip()}")

# Assign a color to each word
unique_words = list(set(words))
cmap = plt.get_cmap("tab10")
color_map = {word: cmap(i % 10) for i, word in enumerate(unique_words)}

# Plot lines for each word (with breaks between different words)
plt.figure(figsize=(12, 5))

start_idx = 0
while start_idx < len(latencies):
    current_word = words[start_idx]
    x_vals = [start_idx]
    y_vals = [latencies[start_idx]]
    
    # Continue while the word is the same
    idx = start_idx + 1
    while idx < len(words) and words[idx] == current_word:
        x_vals.append(idx)
        y_vals.append(latencies[idx])
        idx += 1

    plt.plot(x_vals, y_vals, color=color_map[current_word], label=current_word if start_idx == words.index(current_word) else "")
    
    start_idx = idx

plt.title("Continuous Highlight Latency Line Graph (Colored by Word)")
plt.ylabel("Latency (seconds)")
plt.xticks([])  # 🔹 This removes the tick labels like 50, 100, etc.
plt.grid(True)
plt.legend(title="Word", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.tight_layout()
plt.show()
