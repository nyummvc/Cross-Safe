# ssd
## how to use

put image files and corresponding label files into ./data

default image shape (300, 400, 3) 

label format:   

  class
  
  x1, y1, x2, y2
  
run ./ssd_data_pack.ipynb

run ./ssd_gt_encode.ipynb

run ./ssd_model.ipynb
