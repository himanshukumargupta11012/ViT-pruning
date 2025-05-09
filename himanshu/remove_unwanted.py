import os

path = ""

logs_path = os.path.join(path, "logs")
models_path = os.path.join(path, "models")

print(os.listdir())

for log_file in os.listdir(logs_path):
    with open(os.path.join(logs_path, log_file), "r") as f:
        num_lines = sum(1 for line in f)
    if num_lines < 10:
        os.remove(os.path.join(logs_path, log_file))
        try:
            os.remove(os.path.join(models_path, os.path.splitext(log_file)[0] + ".pth"))
        except:
            print("Model not found for", log_file)
