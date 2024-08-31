% Data
epochs = 1:15;
training_loss = [0.018628128834255003, 0.0053915092448639694, 0.0030384521558856677, ...
                 0.004092779223154974, 0.0033087480922006913, 0.0057532342778235206, ...
                 0.0024615001049017615, 0.0021192899565696467, 0.00461072468863307, ...
                 0.0030610078671304713, 0.005014175834300698, 0.0008138974234032914, ...
                 0.0006710066078619476, 0.0037090211736755625, 0.0001399984868962135];
validation_accuracy = [1.0, 1.0, 1.0, 0.9956573233320174, 1.0, 0.9992104224240032, ...
                       1.0, 0.9984208448480063, 0.9996052112120016, 0.9984208448480063, ...
                       0.9996052112120016, 0.9992104224240032, 0.9996052112120016, ...
                       0.9996052112120016, 1.0];

% Create a new figure
figure;

% Plot Training Loss
yyaxis left;
plot(epochs, training_loss, '-o', 'LineWidth', 2, 'MarkerSize', 6);
ylabel('Training Loss');
ylim([0, max(training_loss) + 0.002]);
xlabel('Epoch');

% Plot Validation Accuracy
yyaxis right;
plot(epochs, validation_accuracy, '-s', 'LineWidth', 2, 'MarkerSize', 6);
ylabel('Validation Accuracy');
ylim([0.99, 1.01]);

% Title and Grid
title('Training Loss and Validation Accuracy over Epochs');
grid on;

% Legend
legend({'Training Loss', 'Validation Accuracy'}, 'Location', 'best');

% Display the figure
