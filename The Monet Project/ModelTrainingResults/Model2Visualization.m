% Load the data from the CSV file with original column headers
data = readtable('results.csv', 'VariableNamingRule', 'preserve');

% Display the column names to verify them
disp(data.Properties.VariableNames);

% Extract the columns
epochs = data.Epoch;
training_loss = data.('Training Loss');
training_accuracy = data.('Training Accuracy');
validation_loss = data.('Validation Loss');
validation_accuracy = data.('Validation Accuracy');

% Create a figure for the plots
figure;

% Plot Training Loss
subplot(2, 2, 1);
plot(epochs, training_loss, '-o');
title('Training Loss');
xlabel('Epoch');
ylabel('Loss');
grid on;

% Plot Training Accuracy
subplot(2, 2, 2);
plot(epochs, training_accuracy, '-o');
title('Training Accuracy');
xlabel('Epoch');
ylabel('Accuracy');
grid on;

% Plot Validation Loss
subplot(2, 2, 3);
plot(epochs, validation_loss, '-o');
title('Validation Loss');
xlabel('Epoch');
ylabel('Loss');
grid on;

% Plot Validation Accuracy
subplot(2, 2, 4);
plot(epochs, validation_accuracy, '-o');
title('Validation Accuracy');
xlabel('Epoch');
ylabel('Accuracy');
grid on;

% Adjust the layout
sgtitle('Training and Validation Metrics');