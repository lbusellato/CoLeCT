from src.recorder import Recorder

def main():
    # Demonstration recording
    recorder = Recorder(server_address='10.85.15.142')
    recorder.run()


if __name__ == '__main__':
    main()