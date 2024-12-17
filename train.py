from models import loader


def train():
    model = loader.ReinforcementLearningModelLoader().load_model()

    print("Model is loaded. Start training...")
    model.train()
    print("Finished!")


if __name__ == "__main__":
    train()