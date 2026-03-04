from openreward.environments import Server

from obscurefacts import ObscureFacts

if __name__ == "__main__":
    server = Server([ObscureFacts])
    server.run()
