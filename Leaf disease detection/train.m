

for i=1:50
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\Brown Spot\',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    Brownspot_Feat3c(i,:) = features(img);
    save Brownspot_Feat3c;
    close all
end


for i=46:49
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\Mosaic\',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    Mosaic_Feat3c(i,:) = features(img);
    save Mosaic_Feat3c;
    close all
end





for i=1:50
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\Powdery Mildew\',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    mildew_Feat3c(i,:) = features(img);
    save mildew_Feat3c;
    close all
end


for i=1:50
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\Scorch\',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    scorch_Feat3c(i,:) = features(img);
    save scorch_Feat3c;
    close all
end

for i=1:25
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\Healthy Images\',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    healthy_Feat3c(i,:) = features(img);
    save healthy_Feat3c;
    close all
end


for i=29:30
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\orange rust\',num2str(i),'.png']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    orange_Feat3c(i,:) = features(img);
    save orange_Feat3c;
    close all
end

for i=31:53
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['training\orange rust\',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    orange_Feat3c(i,:) = features(img);
    save orange_Feat3c;
    close all
end

