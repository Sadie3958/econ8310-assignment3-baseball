# Assignment 3
## Econ 8310 - Business Forecasting

For homework assignment 3, you will work with our baseball pitch data (available in Canvas).

- You must create a custom data loader as described in the first week of neural network lectures to load the baseball videos [2 points]
- You must create a working and trained neural network (any network focused on the baseball pitch videos will do) using only pytorch [2 points]
- You must store your weights and create an import script so that I can evaluate your model without training it [2 points]

Submit your forked repository URL on Canvas! :) I'll be manually grading this assignment.

Some checks you can make on your own:
- Can your custom loader import a new video or set of videos?
- Does your script train a neural network on the assigned data?
- Did your script save your model?
- Do you have separate code to import your model for use after training?

# ADDITIONAL INFORMATION FOR MY ASSIGNMENT
## This is extra stuff in case you would like to know my reasoning on where stuff is and why

- I am using a PyTorch-based neural network to automate the detection process, aiming for the tightest possible bounding boxes around the moving ball.
- Custom Data Loader: I developed a specialized class to handle `.mov` files, extracting frames and converting them into normalized tensors for model ingestion
- Using an employed ResNet-18 architecture, utilizing transfer learning to leverage pre-existing feature recognition capabilities while fine-tuning the final layers for baseball detection
- The baseball_weights folder is in the Releases section (my file was too big to upload as its own file) (44MB oof)
