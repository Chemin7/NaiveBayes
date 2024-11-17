import numpy as np
import pandas as pd


class NaiveBayesClassifier:

    def train(self, feature_data, target_data):
        num_samples, num_features = feature_data.shape
        self.classes = np.unique(target_data)
        num_classes = len(self.classes)

        self.mean = np.zeros((num_classes, num_features), dtype=np.float64)
        self.variance = np.zeros((num_classes, num_features), dtype=np.float64)
        self.priors = np.zeros(num_classes, dtype=np.float64)

        for class_index, class_label in enumerate(self.classes):
            samples_in_class = feature_data[target_data == class_label]
            self.mean[class_index, :] = samples_in_class.mean(axis=0)
            self.variance[class_index, :] = samples_in_class.var(axis=0)
            self.priors[class_index] = samples_in_class.shape[0] / float(num_samples)

    def predict(self, test_data):
        predicted_labels = [self._predict_single(sample) for sample in test_data]
        return np.array(predicted_labels)

    def _predict_single(self, sample):
        posterior_probabilities = []

        for class_index, class_label in enumerate(self.classes):
            log_prior = np.log(self.priors[class_index])
            log_likelihood = np.sum(np.log(self._calculate_pdf(class_index, sample)))
            posterior = log_likelihood + log_prior
            posterior_probabilities.append(posterior)

        return self.classes[np.argmax(posterior_probabilities)]

    def _calculate_pdf(self, class_index, sample):
        class_mean = self.mean[class_index]
        class_variance = self.variance[class_index]
        numerator = np.exp(-((sample - class_mean) ** 2) / (2 * class_variance))
        denominator = np.sqrt(2 * np.pi * class_variance)
        return numerator / denominator


# Pruebas
if __name__ == "__main__":

    dataset = pd.read_excel("C:\\Users\\user\\Downloads\\play.xlsx")

    feature_data = dataset.iloc[:, :-1].values
    target_data = dataset.iloc[:, -1].values

    from sklearn.model_selection import train_test_split

    # Definir función de precisión
    def calculate_accuracy(true_labels, predicted_labels):
        accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
        return accuracy


    while True:
        try:
            num_iterations = int(input("Ingrese el número de iteraciones para evaluar el modelo: "))
            if num_iterations <= 0:
                raise ValueError("El número de iteraciones debe ser un entero positivo.")
            break
        except ValueError as e:
            print(f"Entrada inválida: {e}. Inténtelo de nuevo.")

    accuracies = []

    for i in range(num_iterations):
        train_features, test_features, train_labels, test_labels = train_test_split(
            feature_data, target_data, test_size=0.3, random_state=i
        )

        naive_bayes = NaiveBayesClassifier()
        naive_bayes.train(train_features, train_labels)
        predicted_labels = naive_bayes.predict(test_features)

        accuracy = calculate_accuracy(test_labels, predicted_labels)
        accuracies.append(accuracy)

        print(f"Iteración {i + 1}: Precisión = {accuracy:.4f}")

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print("\nResultados tras varias iteraciones:")
    print(f"Precisión promedio: {mean_accuracy:.4f}")
    print(f"Desviación estándar de la precisión: {std_accuracy:.4f}")
