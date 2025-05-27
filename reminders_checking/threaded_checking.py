import json
import time
import threading
import os

# --- Configuration ---
REMINDERS_FILE = 'C:\\Users\\ADMIN\\Desktop\\Weaselware_Backend\\reminders_queue.json' # Your JSON file name
CHECK_INTERVAL_SECONDS = 1 # How often the daemon wakes up to check/decrement (e.g., every 1 second)

# --- Your Task Trigger Function ---
def trigger_task_reminder(reminder_name, task_description):
    """
    This function is called when a reminder's time runs out.
    Replace this with your actual API call, notification, or other custom logic.
    """
    print(f"\n--- IT'S TIME! ---")
    print(f"Reminder: {reminder_name}")
    print(f"Task: {task_description}")
    # Example: Integrate your API call here:
    # import requests
    # try:
    #     response = requests.post("http://your-api-endpoint.com/trigger", json={"name": reminder_name, "desc": task_description})
    #     response.raise_for_status() # Raise an exception for HTTP errors
    #     print(f"API call successful for {reminder_name}")
    # except requests.exceptions.RequestException as e:
    #     print(f"API call failed for {reminder_name}: {e}")
    print("-" * 30)

# --- The Always-Running Countdown Daemon ---
def reminder_countdown_daemon():
    """
    This function runs continuously in a separate thread.
    It loads reminders, decrements their time, triggers tasks,
    and saves the updated state back to the JSON file.
    """
    print(f" Reminder countdown daemon started. Checking every {CHECK_INTERVAL_SECONDS} second(s).")
    while True: # This loop runs indefinitely
        reminders = []
        try:
            # 1. Load reminders from the JSON file
            if os.path.exists(REMINDERS_FILE) and os.path.getsize(REMINDERS_FILE) > 0:
                with open(REMINDERS_FILE, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list): # Ensure it's a list
                        reminders = loaded_data
                    else:
                        print(f"Warning: {REMINDERS_FILE} content is not a JSON list. Resetting reminders.")
                        reminders = []
            # If file doesn't exist or is empty, reminders list remains empty for this cycle

        except json.JSONDecodeError:
            print(f"⚠️ Error: {REMINDERS_FILE} is corrupted. Cannot load reminders this cycle.")
            reminders = [] # Clear corrupted data to avoid further issues

        updated_reminders = []
        # 2. Process each reminder
        for r in reminders:
            if r.get('active', False): # Only process reminders marked as active
                # Decrement the time
                r['time_in_seconds'] -= CHECK_INTERVAL_SECONDS

                if r['time_in_seconds'] <= 0:
                    # Time's up! Trigger the task and mark as inactive
                    trigger_task_reminder(r['reminder'], f"Task for '{r['reminder']}'")
                    r['active'] = False # Deactivate so it doesn't trigger again
                    r['time_in_seconds'] = 0 # Ensure it doesn't go negative
                    print(f"  - Deactivated: '{r['reminder']}'")
                # Optional: Print ongoing countdown for active reminders (can be noisy)
                # else:
                #     print(f"  - '{r['reminder']}' (Active): {r['time_in_seconds']}s left.")

                updated_reminders.append(r) # Add the processed (updated) reminder to the list
            else:
                updated_reminders.append(r) # Keep inactive reminders in the list

        # 3. Save the updated list back to the JSON file
        try:
            with open(REMINDERS_FILE, 'w') as f:
                json.dump(updated_reminders, f, indent=2)
            # print(f"  - {REMINDERS_FILE} updated successfully.") # Uncomment for verbose logging
        except Exception as e:
            print(f"Error saving {REMINDERS_FILE}: {e}")

        # 4. Pause before the next check cycle
        time.sleep(CHECK_INTERVAL_SECONDS)

# --- Main execution block to start the daemon ---
if __name__ == "__main__":
    print("Starting application...")

    # --- Helper function to add reminders for testing ---
    # This is included here so you can run the whole script as one unit for testing
    def add_reminder_to_queue(reminder_text, initial_time_seconds, filepath=REMINDERS_FILE):
        reminders = []
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        reminders = loaded_data
            except json.JSONDecodeError:
                pass # Will start with empty list if file is corrupted

        new_entry = {
            "reminder": reminder_text,
            "time_in_seconds": initial_time_seconds,
            "active": True
        }
        reminders.append(new_entry)
        with open(filepath, 'w') as f:
            json.dump(reminders, f, indent=2)
        print(f"➕ Added '{reminder_text}' to queue (initial: {initial_time_seconds}s).")

    # --- Initializing Reminders (Optional, for testing) ---
    # Add some test reminders if the file is empty to see the daemon work immediately.
    # You can comment these out if you manage reminders through another part of your app.
    if not os.path.exists(REMINDERS_FILE) or os.path.getsize(REMINDERS_FILE) == 0:
        print(f"No existing reminders in {REMINDERS_FILE}. Adding a few test reminders...")
        add_reminder_to_queue("Short Countdown", 5)  # Triggers in 5 seconds
        add_reminder_to_queue("Medium Task", 15)   # Triggers in 15 seconds
        add_reminder_to_queue("Long Reminder", 30)  # Triggers in 30 seconds

    # --- Start the Daemon Thread ---
    daemon_thread = threading.Thread(target=reminder_countdown_daemon)
    daemon_thread.daemon = True # Allows the main program to exit even if this thread is running
    daemon_thread.start()

    print("\nMain program running. The daemon is working in the background.")
    print("Press Ctrl+C at any time to stop the application.")

    # Keep the main thread alive so the daemon thread can continue running.
    # In a larger application, your main logic would go here (e.g., a web server loop, GUI).
    try:
        while True:
            time.sleep(1) # Sleep briefly to save CPU while main thread waits
    except KeyboardInterrupt:
        print("\n Exiting application. Goodbye!")