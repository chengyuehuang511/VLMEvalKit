import os
import time

src_dir = "/coc/testnvme/chuang475/projects/VLMEvalKit/outputs/GeminiFlash2-0-thinking/T20250521_Gfc0b3333"
dst_dir = "/coc/testnvme/chuang475/projects/VLMEvalKit/outputs/GeminiFlash2-0-thinking"

linked_files = set()

while True:
    try:
        # List all .xlsx files in the source directory
        for fname in os.listdir(src_dir):
            if fname.endswith(".xlsx"):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)

                # Only create symlink if it doesn't exist already
                if fname not in linked_files and not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
                    print(f"Linked: {src_path} -> {dst_path}")
                    linked_files.add(fname)

        time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped by user.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
