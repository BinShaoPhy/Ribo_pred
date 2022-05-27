import os
import numpy as np
import sys

from ribo_model import*
from arguments import*


if __name__ == "__main__":
    
    # parse arguments
    args = get_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    else:
        sys.exit("Error: Folder " + args.out + " already exists")
        
        
    # prepare the datasets
    # sequence.txt contains the 40 AA sequence around the position of interest
    # reads.txt contains the Ribo-seq read depth at the position of interest
    x = np.loadtxt('sequence.txt', delimiter="\t")
    y = np.loadtxt('reads.txt', delimiter="\t")

    num_val_samples = int(0.15 * x.shape[0])
    num_train_samples = x.shape[0] - 2 * num_val_samples
    indices = np.random.permutation(x.shape[0])

    training_idx, test_idx = indices[:num_train_samples], indices[num_train_samples:num_train_samples + num_val_samples]
    val_idx = indices[num_train_samples + num_val_samples:]

    (x_train, y_train) = (x[training_idx], y[training_idx])
    (x_val, y_val) = (x[val_idx], y[val_idx])
    (x_test, y_test) = (x[test_idx], y[test_idx])
    
    
    # load keras model
    print("-------------------------------------")
    print("             Training Model          ")
    print("-------------------------------------")
    
    DTModel = DeepTransModel(
        learning_rate = args.learning,
        num_heads = args.nheads,
        layer_num = args.nconv,
    )
    if args.verbose:
        with_verbose = 1
    else:
        with_verbose = 0
        
    history = DTModel.train(x_train, y_train, n_epochs = args.epoch)
    
    
    # evaluate model and save results
    print("-------------------------------------")
    print("           Evaluating Model          ")
    print("-------------------------------------")
    
    loss, train_acc = DTModel.evaluate(
          x, y, batch_size = 250, verbose = 2
    )
    
    np.save( args.out + "/history.npy", history.history)
                
    DTModel.save( args.out + '/model.h5')
    with open( args.out + "/model_summary.txt", "w") as f:
        f.write(DTModel.to_json())
        f.write("\nTraining Loss: " + str(loss) + "\n")

   