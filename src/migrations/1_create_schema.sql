CREATE TABLE IF NOT EXISTS migrations (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) UNIQUE NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS heart_disease (
    id SERIAL PRIMARY KEY,
    age INT NOT NULL,
    sex INT NOT NULL,
    cp INT NOT NULL,
    trestbps INT NOT NULL,
    chol INT NOT NULL,
    fbs INT NOT NULL,
    restecg INT NOT NULL,
    thalach INT NOT NULL,
    exang INT NOT NULL,
    oldpeak FLOAT NOT NULL,
    slope INT NOT NULL,
    ca INT NOT NULL,
    thal INT NOT NULL,
    target INT NOT NULL
);
