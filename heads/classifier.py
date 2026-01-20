import paddle.nn as nn

class ArcFaceClassifier(nn.Layer):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings):
        return self.fc(embeddings)
