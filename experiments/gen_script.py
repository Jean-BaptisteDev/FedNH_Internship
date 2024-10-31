import os  # Import the os module for operating system dependent functionality
import platform  # Import the platform module to access system information
import glob  # Import the glob module for filename pattern matching

def gen_command(
        purpose,  # Purpose of the command, e.g., the dataset or task
        device,  # Device to run the command on (e.g., CUDA or CPU)
        global_seed,  # Global seed for reproducibility
        use_wandb,  # Flag to indicate whether to use Weights & Biases for logging
        yamlfile,  # Path to the YAML configuration file
        strategy,  # Federated learning strategy to use (e.g., FedAvg, FedNH)
        num_clients,  # Number of clients participating in federated learning
        participate_ratio,  # Ratio of clients participating in each round
        partition,  # Type of data partitioning strategy
        partition_arg,  # Argument for the partitioning strategy
        partition_val,  # Value for the partitioning argument
        num_rounds,  # Number of training rounds
        client_lr,  # Learning rate for the clients
        client_lr_scheduler,  # Learning rate scheduler for the clients
        sgd_momentum,  # Momentum value for SGD
        sgd_weight_decay,  # Weight decay for regularization in SGD
        num_epochs,  # Number of epochs for training
        **kwargs):  # Additional keyword arguments for hyperparameters

    # Construct the command string with all specified parameters
    command = f"python3 ../main.py  --purpose {purpose} --device {device} --global_seed {global_seed} --use_wandb {use_wandb} --yamlfile {yamlfile} --strategy {strategy} --num_clients {num_clients} --participate_ratio {participate_ratio} --partition {partition} --{partition_arg} {partition_val} --num_rounds {num_rounds} --client_lr {client_lr} --client_lr_scheduler {client_lr_scheduler} --sgd_momentum {sgd_momentum} --sgd_weight_decay {sgd_weight_decay} --num_epochs {num_epochs}"

    command_hyper = ""  # Initialize an empty string for additional hyperparameters
    for k, v in kwargs.items():  # Loop through each additional hyperparameter
        command_hyper += f" --{k} {v}"  # Append the hyperparameter to the command
    command += command_hyper + " &"  # Add the hyperparameters to the command and run it in the background
    return command  # Return the constructed command

if __name__ == "__main__":  # Check if the script is run as the main program
    print(f"Current working directory: {os.getcwd()}")  # Print the current working directory

    # Delete .sh files in a cross-platform way
    for filename in glob.glob('*.sh'):  # Loop through all .sh files in the current directory
        try:
            os.remove(filename)  # Attempt to remove the file
            print(f"Deleted file: {filename}")  # Print confirmation of deletion
        except OSError as e:  # Catch any error during file deletion
            print(f"Error deleting file {filename}: {e}")  # Print the error message

    # Set the purpose for the current run
    purpose = 'Cifar'  # You can uncomment the line below for different purposes
    # purpose = 'TinyImageNet'
    # purpose = 'BrainTumorMRI'
    
    num_gpu = 4  # Set the number of GPUs available for training
    if purpose == 'TinyImageNet':  # Check if the purpose is TinyImageNet
        # Define the strategies and their corresponding hyperparameters
        strategy_hyper = [('FedAvg', {'no_norm': False}), ('FedNH', {'no_norm': True}),
                          ('FedROD', {'FedROD_hyper_clf': False, 'FedROD_phead_separate': True, 'no_norm': False}), 
                          ('FedProto', {'no_norm': False}),
                          ('FedRep', {'no_norm': False}), ('FedBABU', {'FedBABU_finetune_epoch': 5, 'no_norm': False}),
                          ('FedPer', {'no_norm': False}), ('Ditto', {'Ditto_lambda': 0.75, 'no_norm': False}),
                          ('Local', {'no_norm': False}),
                          ('CReFF', {'CReFF_lr_feature': 0.01, 'CReFF_lr_net': 0.01})]
        
        planned, actual = 0, 0  # Initialize planned and actual counts for commands
        yamlfile_lst = ['./TinyImageNet_ResNet.yaml']  # List of YAML files for configuration
        num_round = 200  # Number of training rounds
        client_lr = 0.01  # Learning rate for the clients
        client_lr_scheduler = 'diminishing'  # Learning rate scheduler type
        sgd_momentum = 0.9  # Momentum for SGD
        sgd_weight_decay = 0.001  # Weight decay for SGD
        num_epochs = 5  # Number of epochs to train

        # Loop through different run types and YAML files to generate commands
        for run_type in ['beta']:
            for yamlfile in yamlfile_lst:
                for strategy, hyper in strategy_hyper:
                    if run_type == 'beta':
                        for beta in ['0.3', '1.0']:  # Loop through beta values
                            for pratio in [0.1]:  # Loop through participation ratios
                                cuda = f'cuda:{planned % num_gpu}'  # Assign GPU based on planned count
                                # Generate command based on presence of hyperparameters
                                if hyper is not None:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs, **hyper)
                                else:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs)

                                planned += 1  # Increment planned count
                                if command is not None:  # Check if command is generated
                                    actual += 1  # Increment actual count
                                    filename = f'{strategy}_dir.sh'  # Define filename for the command script
                                    print(f"Writing command to {filename}")  # Inform about command writing
                                    # Open file in binary mode and use LF line endings
                                    with open(filename, 'ab') as f:
                                        f.write(command.encode('utf-8') + b'\n')  # Write command to the file
                                    print(f"Created file: {filename}")  # Confirm file creation

        print(f"actual/planned:{actual}/{planned}")  # Print summary of commands created

    if purpose == 'Cifar':  # Check if the purpose is Cifar
        # Define strategies and hyperparameters for Cifar
        strategy_hyper = [('FedAvg', None), ('FedNH', None),
                          ('FedROD', {'FedROD_phead_separate': True}), ('FedProto', None),
                          ('FedRep', None), ('FedBABU', {'FedBABU_finetune_epoch': 5}),
                          ('FedPer', None), ('Ditto', {'Ditto_lambda': 0.75}),
                          ('Local', {'no_norm': False}),
                          ('CReFF', {'CReFF_lr_feature': 0.01, 'CReFF_lr_net': 0.01})]
        
        yamlfile_lst = ['./Cifar10_Conv2Cifar.yaml', './Cifar100_Conv2Cifar.yaml']  # List of YAML files for Cifar
        planned, actual = 0, 0  # Reset planned and actual counts
        num_round = 200  # Number of training rounds
        client_lr = 0.01  # Learning rate for clients
        client_lr_scheduler = 'diminishing'  # Learning rate scheduler type
        sgd_momentum = 0.9  # Momentum for SGD
        sgd_weight_decay = 0.00001  # Weight decay for SGD
        num_epochs = 5  # Number of training epochs
        
        # Loop through different run types and YAML files to generate commands
        for run_type in ['beta']:
            for yamlfile in yamlfile_lst:
                for strategy, hyper in strategy_hyper:
                    if run_type == 'beta':
                        for beta in ['0.3', '1.0']:  # Loop through beta values
                            for pratio in [0.1]:  # Loop through participation ratios
                                cuda = f'cuda:{planned % num_gpu}'  # Assign GPU based on planned count
                                # Generate command based on presence of hyperparameters
                                if hyper is not None:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs, **hyper)
                                else:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs)

                                planned += 1  # Increment planned count
                                if command is not None:  # Check if command is generated
                                    actual += 1  # Increment actual count
                                    filename = f'{strategy}_dir.sh'  # Define filename for the command script
                                    print(f"Writing command to {filename}")  # Inform about command writing
                                    # Open file in binary mode and use LF line endings
                                    with open(filename, 'ab') as f:
                                        f.write(command.encode('utf-8') + b'\n')  # Write command to the file
                                    print(f"Created file: {filename}")  # Confirm file creation

        print(f"actual/planned:{actual}/{planned}")  # Print summary of commands created
