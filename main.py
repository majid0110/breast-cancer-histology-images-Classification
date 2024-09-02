from data import load_data
from model import build_glnet_model, train_model, evaluate_model
from predict import predict_image, plot_predictions, test_path

def main():
    X_train, X_test, y_train, y_test = load_data()

    input_shape = (250, 200, 3)
    glnet_model = build_glnet_model(input_shape)
    train_model(glnet_model, X_train, y_train, X_test, y_test)
    evaluate_model(glnet_model, X_test, y_test)
    example_image_path = os.path.join(test_path, 'sample_image.tif')
    predicted_class, confidence = predict_image(glnet_model, example_image_path)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

    plot_predictions(glnet_model, test_path, num_images=5)

if __name__ == "__main__":
    main()
