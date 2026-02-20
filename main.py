from src.engine import GraphicsEngine

if __name__ == '__main__':
    # Ensure you have a 'assets/textures/test.png' or the fallback will run
    app = GraphicsEngine()
    app.run()