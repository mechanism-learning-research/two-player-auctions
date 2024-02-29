import sqlite3
import json
import re
from tabulate import tabulate
import jax
import jax.numpy as jnp
import pickle  # Assuming the content is pickled

# Connect to the database
conn = sqlite3.connect("results.db")
cursor = conn.cursor()

# Execute the SQL query for runs along with the regret data
cursor.execute(
    """
    SELECT run.run_id, run.captured_out, JSON_EXTRACT(run.config, '$') AS config, 
           artifact.content
    FROM run
    JOIN experiments_sources ON run.experiment_id = experiments_sources.experiment_id
    LEFT JOIN artifact ON run.run_id = artifact.run_id AND artifact.filename = 'regret'
    WHERE run.status = 'COMPLETED'
"""
)
config_keys = [
    "num_steps",
    "misr_reinit_iv",
    "misr_reinit_lim",
    "batch_size",
    "net_width",
    "net_depth",
    "learning_rate",
    "rng_seed_training",
    "rng_seed_test",
]

headers = ["Bidders", "Items", "Score", "Regret", "Revenue"] + config_keys
rows = []

for row in cursor.fetchall():
    run_id, captured_out, config, regret_content = row

    # Parse the config as a dictionary
    config = json.loads(config)

    # Deserialize and process the regret data
    if regret_content:
        regret_matrix = pickle.loads(regret_content)
        regret_matrix = jax.nn.relu(regret_matrix)
        total_values = jnp.sum(regret_matrix, axis=1)
        average_total_value = jnp.mean(total_values)
    else:
        average_total_value = None

    lines = captured_out.split("\n")
    for line in lines:
        if line.startswith("pay:"):
            pay = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            # Use the computed average_total_value as regret
            regret = average_total_value if average_total_value is not None else 0
            model_score = (pay**0.5) - max(0, regret) ** 0.5

            # Add the data to the rows list in markdown table format
            config_data = [config[key] for key in config_keys]
            if config["misr_updates"] == 100 and config["num_test_samples"] == 10000:
                rows.append(
                    [config["bidders"], config["items"], model_score, regret, pay]
                    + config_data
                )

# Sort the rows by "bidders", "items" in ascending order, and "model score" in descending order
rows.sort(key=lambda row: (row[0], row[1], -row[2]))
# Print the result as a markdown table
print(tabulate(rows, headers=headers, tablefmt="pipe"))

# Close the connection
conn.close()
