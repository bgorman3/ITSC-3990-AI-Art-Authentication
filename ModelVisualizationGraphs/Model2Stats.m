% Data
epochs = 1:15;
training_loss = [0.037075640289259075, 0.023121547504855622, 0.009102630887392531, ...
                 0.0062600187119445806, 0.017059910570936483, 0.004961307863086217, ...
                 0.007369875655965403, 0.0029910543275319555, 0.0058477008128620936, ...
                 0.006733279305821679, 0.006150470763824342, 0.004799543860399177, ...
                 0.002693762433411729, 0.007400051022904506, 0.012894413672444061];
validation_accuracy = [0.9913146466640348, 0.6533754441373865, 0.9996052112120016, ...
                       0.9976312672720095, 0.9988156336360048, 1.0, ...
                       1.0, 1.0, 0.9996052112120016, 0.9980260560600079, ...
                       1.0, 0.9996052112120016, 0.9968416896960126, 1.0, ...
                       0.9976312672720095];

% Create a new figure
figure;

% Plot Training Loss
yyaxis left;
plot(epochs, training_loss, '-o', 'LineWidth', 2, 'MarkerSize', 6);
ylabel('Training Loss');
ylim([0, max(training_loss) + 0.002]);  % Adjust y-axis for Training Loss
xlabel('Epoch');

% Plot Validation Accuracy
yyaxis right;
plot(epochs, validation_accuracy, '-s', 'LineWidth', 2, 'MarkerSize', 6);
ylabel('Validation Accuracy');
ylim([0.65, 1.01]);  % Adjust y-axis for Validation Accuracy
xlabel('Epoch');

% Title and Grid
title('Training Loss and Validation Accuracy over Epochs');
grid on;

% Legend
legend({'Training Loss', 'Validation Accuracy'}, 'Location', 'best');

% Display the figure
