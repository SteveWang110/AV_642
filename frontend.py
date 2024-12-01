import tkinter as tk
import subprocess
from tkinter import ttk
from googletrans import Translator
from googletrans import LANGUAGES
from carla_lane_keeping_d3qn import update_plot
import subprocess
import threading
import csv

# import carla

# Global variables to store rewards and num_steps
rewards = []
num_steps = []

language_dict = {name: code for code, name in LANGUAGES.items()}


def translate_labels(root, dest_language):
    """
    Translate labels in the Frontend GUI to the specified destination language.

    Args:
        root (tk.Tk): The root window of the GUI.
        dest_language (str): The destination language to translate to.
    """
    translator = Translator()
    print(dest_language)
    dest_language_code = language_dict[dest_language]

    for widget in root.winfo_children():
        if isinstance(widget, (tk.Label, tk.Button)):
            original_text = widget.cget("text")
            translated_text = translator.translate(
                original_text, dest=dest_language_code
            ).text
            widget.config(text=translated_text)


# available operations
available_ops = ["New", "Load", "Tune"]

# reward functions
available_rewards = ["1", "2", "3", "4"]

# available maps
available_maps = ["Town01", "Town02", "Town03", "Town04", "Town05"]
true_false = ["True", "False"]


def show_plot():
    
    """
        Display the plot of rewards versus number of steps, lane deviation, angle, and speed.
    """

    try:
        csv_file = "plot_data.csv"

        file = open(csv_file, "r", newline="")
        reader = csv.reader(file)
        rewards = []
        num_steps = []
        for row in reader:
            x, y = map(float, row)
            rewards.append(x)
            num_steps.append(y)

        file.close()

        csv_file2 = "step_plot.csv"
        file2 = open(csv_file2, "r", newline="")
        reader2 = csv.reader(file2)
        lane_deviation = []
        speed = []
        angle = []

        next(reader2)
        for row in reader2:
            x, y, z = map(float, row)
            lane_deviation.append(x)
            angle.append(y)
            speed.append(z)

        file2.close()
    except StopIteration:
        print("The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

    update_plot(rewards, num_steps, lane_deviation, angle, speed)


def run_backend():
    """
    Run the backend script with the provided arguments.
    """
    run_button.config(state="disabled")
    arg1 = entry1.get()
    arg2 = entry2.get()
    arg3 = entry3.get()
    arg4 = entry4.get()
    arg5 = entry5.get()
    arg6 = entry6.get()
    arg7 = entry7.get()
    arg8 = entry8.get()
    arg9 = entry9.get()

    subprocess.run(
        [
            "python",
            "carla_lane_keeping_d3qn.py",
            "--version",
            arg1,
            "--operation",
            arg2,
            "--save-path",
            arg3,
            "--reward-function",
            arg4,
            "--map",
            arg5,
            "--epsilon-decrement",
            arg6,
            "--num-episodes",
            arg7,
            "--max-steps",
            arg8,
            "--random-spawn",
            arg9,
        ]
    )
    run_button.config(state="normal")


def run_backend_thread():
    """
    Run the backend script in a separate thread.
    """
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.start()


root = tk.Tk()
root.title("CARLA User Interface")

style = ttk.Style(root)
style.theme_use("classic")

# Create input fields
label1 = tk.Label(root, text="Version:")
label1.grid(row=0, column=0, padx=5, pady=5)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=5, pady=5)

label2 = tk.Label(root, text="Operation:")
label2.grid(row=1, column=0, padx=5, pady=5)
entry2 = ttk.Combobox(root, values=available_ops)
entry2.grid(row=1, column=1, padx=5, pady=5)

label3 = tk.Label(root, text="Save Path:")
label3.grid(row=2, column=0, padx=5, pady=5)
entry3 = tk.Entry(root)
entry3.grid(row=2, column=1, padx=5, pady=5)

label4 = tk.Label(root, text="Reward Function:")
label4.grid(row=3, column=0, padx=5, pady=5)
entry4 = ttk.Combobox(root, values=available_rewards)
entry4.grid(row=3, column=1, padx=5, pady=5)

label5 = tk.Label(root, text="Map:")
label5.grid(row=4, column=0, padx=5, pady=5)
entry5 = ttk.Combobox(root, values=available_maps)
entry5.grid(row=4, column=1, padx=5, pady=5)

label6 = tk.Label(root, text="Epsilon Decrement:")
label6.grid(row=5, column=0, padx=5, pady=5)
entry6 = tk.Entry(root)
entry6.insert(0, "0.005")
entry6.grid(row=5, column=1, padx=5, pady=5)

label7 = tk.Label(root, text="Num Episodes:")
label7.grid(row=6, column=0, padx=5, pady=5)
entry7 = tk.Entry(root)
entry7.insert(0, "600")
entry7.grid(row=6, column=1, padx=5, pady=5)

label8 = tk.Label(root, text="Max Num Steps:")
label8.grid(row=7, column=0, padx=5, pady=5)
entry8 = tk.Entry(root)
entry8.insert(0, "300")
entry8.grid(row=7, column=1, padx=5, pady=5)

label9 = tk.Label(root, text="Random Vehicle Spawn Location?")
label9.grid(row=8, column=0, padx=5, pady=5)
entry9 = ttk.Combobox(root, values=true_false)
entry9.insert(0, "True")
entry9.grid(row=8, column=1, padx=5, pady=5)


# Create language dropdown menu
languages = list(LANGUAGES.values())
language_select_label = tk.Label(root, text="Select Language:")
language_select_label.grid(row=9, column=0, padx=5, pady=5)
# languages = [ 'fr', 'en', 'sp']  # Add more languages as needed
language_select = ttk.Combobox(root, values=languages)

language_select.grid(row=9, column=1, padx=5, pady=5)

language_select.current(21)  # Set default language
language = language_select.current(21)  # Set default language
# root.language = language_select.get()
# Create 'Translate' button
translate_button = tk.Button(
    root,
    text="Translate",
    command=lambda: translate_labels(root, language_select.get()),
)
translate_button.grid(row=9, column=2, columnspan=2, padx=(20, 20), pady=5)


# Create 'Run' button
run_button = tk.Button(root, text="Run Backend", command=run_backend_thread)
run_button.grid(row=10, column=0, columnspan=2, padx=5, pady=5)

plot_button = tk.Button(root, text="Show Plot", command=show_plot)
plot_button.grid(row=10, column=1, columnspan=2, padx=(20, 20), pady=5)


# Start the GUI event loop
root.mainloop()
