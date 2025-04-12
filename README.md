# Research and Development of a Product Recommendation System using Deep Matrix Factorization

## Introduction

The project "Research and Development of a Product Recommendation System using Deep Matrix Factorization" is a graduation internship project aimed at building an intelligent recommendation system that combines various modern Machine Learning methods. This system uses Deep Matrix Factorization (DMF) as its main foundation, combined with K-Nearest Neighbors (KNN) and Popularity-Based Model to create a personalized and accurate product recommendation experience.

In the context of e-commerce, recommending suitable products helps users save search time and increases the likelihood of finding desirable products. The system is designed to efficiently handle challenges such as the cold-start problem (new users), real-time updates, and deep personalization based on user behavior.

## Key Features

- **Personalized Recommendations**: Using the Deep Matrix Factorization model to learn non-linear representations of users and products
- **Cold-start Handling**: Combining Popularity-Based Model to recommend popular products for new users
- **Similarity Calculation**: Using KNN and cosine similarity to find the closest products in the feature space
- **Incremental Learning**: Updating the model in real-time when new user interactions occur
- **User Clustering**: Applying the K-Means algorithm to group users according to similar behaviors
- **MongoDB Integration**: Efficiently storing and managing user interaction data
- **Personalization**: Combining category information, keywords, and search history to increase recommendation accuracy
- **User-friendly Web Interface**: Built with Flask, providing a good user experience with responsive design

## Technologies Used

- **Python 3.8+**: Main programming language
- **TensorFlow 2.x/Keras**: Building and training the Deep Matrix Factorization model
- **Scikit-learn**: Implementing KNN, K-Means algorithms and data processing
- **MongoDB**: NoSQL database for storing user information, products, and interactions
- **Flask**: Lightweight Python web framework for building the user interface
- **Pandas/NumPy**: Data processing and analysis libraries
- **Bootstrap**: CSS framework for responsive web design
- **Matplotlib/Seaborn**: Data visualization libraries
- **Bcrypt**: Password encryption
- **Jupyter Notebook**: Development and model testing environment

## Project Structure

```
N47_HeThongGoiYSanPham/
│
├── data/
│   ├── __pycache__/                # Python cache
│   └── database.py                 # Module for MongoDB connection and interaction
│
├── static/
│   ├── css/
│   │   └── style.css               # Custom CSS for web interface
│   ├── img/
│   │   └── placeholder.png         # Default image for products without images
│   └── js/
│       └── main.js                 # JavaScript handling user interactions
│
├── templates/
│   ├── auth/                       # Templates for authentication
│   │   ├── login.html              # Login page
│   │   └── signup.html             # Signup page
│   ├── cart.html                   # Shopping cart page
│   ├── category.html               # Products by category page
│   ├── home.html                   # Homepage
│   ├── layout.html                 # Base template for all pages
│   ├── product_detail.html         # Product details page
│   ├── require_login.html          # Login requirement page
│   └── search_results.html         # Search results page
│
├── utils/
│   ├── __pycache__/                # Python cache
│   ├── __init__.py                 # Utils package initialization
│   ├── data_processing.py          # Data processing module
│   └── recommendation.py           # Recommendation logic module
│
├── amazon_cleaned.csv              # Cleaned and balanced Amazon data
├── amazon_recommender_metadata_updated.pkl  # Model metadata
├── amazon_recommender_model_updated.keras   # Trained DMF model
│
├── app.py                          # Main Flask application
│
├── best_model.keras                # Best model saved from training process
│
├── cart_categories_cntt@gmail.com.json     # User's cart categories
│
├── clean_data.ipynb                # Notebook for data cleaning and balancing
│
├── example.py                      # Example file for creating sample metadata
│
├── Link amazon reviews 2023.txt          # Link to Amazon Reviews 2023 dataset
├── Link amazon sales dataset (16 columns).txt  # Link to Amazon Sales Dataset from Kaggle
├── Link kết nối MongoDB.txt              # MongoDB Compass connection link
│
├── MongoDB và Fine-Tuning.ipynb    # Notebook for incremental learning system
│
├── product_embeddings.npy          # File storing product embedding vectors
│
├── True First DMF.ipynb            # Notebook implementing Deep Matrix Factorization model
│
└── user_embeddings.npy             # File storing user embedding vectors
```

## Models Used

### 1. Deep Matrix Factorization (DMF)
![Deep Matrix Factorization Model Architecture](dmf_architecture.jpg)
- **Architecture**: Deep neural network with 32-dimensional embedding layers for both users and products, followed by fully connected (dense) layers with decreasing sizes 256→128→64→1
- **Regularization**: Using L2 regularization (0.01) for embedding layers
- **Batch Normalization**: Applied after each dense layer to stabilize the training process
- **Activation**: Using LeakyReLU (alpha=0.0001) instead of ReLU to avoid the "dying neurons" problem
- **Dropout**: 0.2 rate applied to the first two dense layers to reduce overfitting
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with low learning rate (0.00001) and clipnorm=1.0

### 2. K-Nearest Neighbors (KNN)
- Used from scikit-learn library with parameter metric="cosine"
- Finding products with the closest embedding vectors in the feature space
- Combining multiple factors when calculating the composite score:
  - Cosine similarity (30%)
  - Popularity score (10%)
  - Category score (20-30%)
  - Keyword score (30%)

### 3. Popularity-Based Model
- Ranking products based on the product of the number of ratings and the average rating
- Normalizing popularity scores to the range [0, 1]
- Primarily used for new users or when data is lacking

### 4. K-Means Algorithm
- Clustering users or products based on embedding vectors
- Used to analyze behavior and create insights from data

## Product Recommendation Process

1. **New Users**:
   - Using Popularity-Based Model to recommend the most popular products
   - If information about search keywords or categories of interest is available, the system combines it with popularity scores

2. **Users with History**:
   - Collecting information from multiple sources (recent search keywords, viewed product categories)
   - Using the user's embedding vector (learned from the Deep Matrix Factorization model)
   - Combining KNN to find the most similar products
   - Calculating composite scores based on multiple criteria
   - Providing a list of the most suitable product recommendations

3. **Model Updates**:
   - **Batch Fine-Tuning**: Periodically updating the model with newly accumulated interaction data
   - **Real-time Embedding Updates**: Immediately updating embedding vectors based on each new user interaction
   - **Embedding Layer Expansion**: Automatically expanding the size of the embedding layer when new users or products are added

## Installation and Usage

### Requirements

- Python 3.8 to 3.12
- MongoDB
- TensorFlow 2.x

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/N47_HeThongGoiYSanPham.git
cd N47_HeThongGoiYSanPham
```

2. Configure MongoDB Compass connection:
   - Use the connection string:
   ```
   mongodb+srv://nhoktk000:Quan26012004@cluster.ap9ga.mongodb.net/ecommerce_db?retryWrites=true&w=majority
   ```

3. Start the application:
```bash
python app.py
```

### Training Models from Scratch

1. Process and clean data:
   - Run the notebook `clean_data.ipynb`

2. Train the DMF model:
   - Run the notebook `True First DMF.ipynb`

3. Configure incremental learning:
   - Run the notebook `MongoDB và Fine-Tuning.ipynb`

### Using Pre-trained Models

1. Ensure the following files exist:
   - `amazon_recommender_model_updated.keras`
   - `amazon_recommender_metadata_updated.pkl`
  
or:
   - `amazon_recommender_model.keras`
   - `amazon_recommender_metadata.pkl`
These are two trained model files for the True First DMF.ipynb file

2. Start the web application:
```bash
python app.py
```

3. Access the web application via a browser:
```
http://localhost:5000
```

## Usage Guide

1. **Register/Login**: Create a new account or log in to the system
2. **View Products**: Browse the product list or search by keywords
3. **Personalized Recommendations**: View products recommended based on your interaction history
4. **Shopping Cart**: Add products to cart and view the total
5. **Search**: Find products by keywords, name, or category

## Detailed Documentation

### API Endpoints

- **GET /**: Homepage
- **GET /product/<product_id>**: Product details
- **GET /category/<category>**: Products by category
- **GET /search?q=<query>**: Search products
- **GET /cart**: View cart
- **POST /add_to_cart/<product_id>**: Add product to cart
- **GET /remove_from_cart/<product_id>**: Remove product from cart
- **POST /update_cart**: Update product quantity in cart
- **GET /login**: Login page
- **POST /login**: Process login
- **GET /signup**: Signup page
- **POST /signup**: Process signup
- **GET /logout**: Logout

### MongoDB Database Structure

- **Collection users**: Store user information
  - username: Username
  - email: Email (used as login ID)
  - password: Encrypted password
  - created_at: Account creation time
  - cart_history: Cart history
  - view_history: Product viewing history

- **Collection user_interactions**: Store user-product interactions
  - user_id: User ID
  - product_id: Product ID
  - interaction_type: Interaction type (view, add_to_cart, purchase, rate)
  - rating: Rating (if any)
  - timestamp: Interaction time

- **Collection products**: Store product information
  - product_id: Product ID
  - product_name: Product name
  - keywords: Related keywords
  - category: Product category
  - created_at: Creation time

- **Collection embeddings**: Store embedding vectors
  - type: Type (user or product)
  - user_id/product_id: User ID/Product ID
  - embedding: Embedding vector

- **Collection browsing_history**: Store web browsing history
  - user_id: User ID
  - product_id: Product ID
  - timestamp: Viewing time
  - category: Product category (if any)

- **Collection metadata_collection**: Store system metadata
  - key: Metadata key
  - value: Corresponding value

## Authors

- **[Group 47 - IT Graduation Internship - FitVAA]**

## License

This project is distributed under the MIT license. See the `LICENSE` file for more details.

## Acknowledgements

- Thanks to **[Dr. Tran Nguyen Bao]** for support and guidance

## References

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Keras Documentation](https://keras.io/guides/)
- [Deep Matrix Factorization with TensorFlow](https://st1990.hatenablog.com/entry/2019/03/16/231554)
- [Amazon Sales Dataset on Kaggle](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- [Building Recommendation System Using KNN](https://aurigait.com/blog/recommendation-system-using-knn/)
- [What Are Recommendation Systems in Machine Learning](https://www.analyticssteps.com/blogs/what-are-recommendation-systems-machine-learning)
- [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/)
