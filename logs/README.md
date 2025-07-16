In our first attempt we have tried with using `Harversine Distance Loss` which is computing the radius of Earth and taking account of the relative distances of the GPS from the Ground Truth. However, due to the linearity of the loss function and not abel to compute 2D distances, we recieved a regression model for only computing the least route to each location: 
<img width="952" height="849" alt="image" src="https://github.com/user-attachments/assets/6cd3807a-7c47-4543-88f6-7fbe5f589a21" />

We have also tinkered with Depth Maps, hoping that it would be helpful for predcition the ground truth location, and when we only compute upon the depth maps we recieve:
<img width="787" height="821" alt="image" src="https://github.com/user-attachments/assets/b92270ef-2c71-4cb7-bf59-e99fce03d45b" />
