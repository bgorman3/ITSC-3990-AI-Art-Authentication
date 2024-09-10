% Data
epochs = 1:15;
training_loss = [0.5160977547388197, 0.46804566251184154, 0.4590177830293089, ...
                 0.39933928448286016, 0.33262272696614764, 0.2848442021024776, ...
                 0.4135631302909372, 0.31368568190470897, 0.30527760281722416, ...
                 0.25851401300111077, 0.25397059371281866, 0.25234852768636645, ...
                 0.20973107443195008, 0.15842528048918336, 0.1404980982821357];
validation_accuracy = [0.5, 0.5294117647058824, 0.6764705882352942, 0.7941176470588235, ...
                       0.7941176470588235, 0.75, 0.7647058823529411, 0.7647058823529411, ...
                       0.8088235294117647, 0.7352941176470589, 0.6911764705882353, ...
                       0.8676470588235294, 0.8235294117647058, 0.8823529411764706, ...
                       0.8088235294117647];

% Create a new figure
figure;

% Plot Training Loss
yyaxis left;
plot(epochs, training_loss, '-o', 'LineWidth', 2, 'MarkerSize', 6);
ylabel('Training Loss');
ylim([0, max(training_loss) + 0.1]);
xlabel('Epoch');

% Plot Validation Accuracy
yyaxis right;
plot(epochs, validation_accuracy, '-s', 'LineWidth', 2, 'MarkerSize', 6);
ylabel('Validation Accuracy');
ylim([0.4, 1.0]);

% Title and Grid
title('Training Loss and Validation Accuracy over Epochs');
grid on;

% Legend
legend({'Training Loss', 'Validation Accuracy'}, 'Location', 'best');

% Display the figure