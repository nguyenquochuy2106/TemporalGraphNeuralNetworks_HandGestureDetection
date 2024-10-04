# main.py
import install_dependencies  # Ensure to run this first
import data_collection
import labeling
import model_training
import streamlit_app

def main():
    # Install required libraries
    install_dependencies.install_packages()
    # Uncomment the following lines to run specific tasks:
    # data_collection.collect_images()  # Run image collection
    # labeling.label_images()            # Run labeling
    # model_training.train_model()       # Run training
    # streamlit_app.main()               # Run Streamlit application
if __name__ == "__main__":
    main()
