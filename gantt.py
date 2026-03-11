import matplotlib.pyplot as plt

tasks = [
    ("Requirement Analysis", 1, 1),
    ("System Design", 2, 2),
    ("Dataset Collection", 4, 1),
    ("Feature Extraction", 5, 2),
    ("ML Training", 7, 2),
    ("Integration", 9, 2),
    ("Testing", 11, 1),
    ("Documentation", 12, 1),
]

fig, ax = plt.subplots(figsize=(10, 6))

for task in tasks:
    ax.barh(task[0], task[2], left=task[1])

# Make first task appear at top
ax.invert_yaxis()

# Set X-axis range
ax.set_xlim(0, 14)

# Show all week numbers
ax.set_xticks(range(0, 14, 1))   # 0 to 13

ax.set_xlabel("Weeks")
ax.set_ylabel("Tasks")
ax.set_title("Project Gantt Chart")

ax.grid(True, linestyle='--', alpha=0.5)

plt.show()