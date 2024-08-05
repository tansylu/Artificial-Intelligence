import matplotlib.pyplot as plt

# Read the results from the text file
with open('best_results.txt', 'r') as file:
    results = [float(line.strip()) for line in file]

# Create a list of generation numbers
generations = list(range(1, len(results) + 1))

# Create the plot
plt.plot(generations, results)
plt.xlabel('Generation')
plt.ylabel('Best Result')
plt.title('Best Result vs Generation')
plt.grid(True)

# Show the plot
plt.show()