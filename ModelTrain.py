from ultralytics import YOLO


# TRAIN ON A COMPUTER WITH A HEFTY GRAPHICS CARD, A LAPTOP WILL TAKE A WHOLE DAY TO TRAIN THE YOLO MODEL!!!


# Load a pre-trained model
model = YOLO('yolo26n.pt')  # You can find pre-trained weights on the Ultralytics documentation

# Train the model
results = model.train(data='Yolo_Data\data.yaml', epochs=50, imgsz=640, batch=16)

print(model.names)