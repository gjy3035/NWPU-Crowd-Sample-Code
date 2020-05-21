%%
clc; clear all;
init_maxSize = [2048,2048];
minSize = [576,768];

path ='path to the data path';
output_path = strcat('../NWPU-Crowd/min_', num2str(minSize(1)), 'x', num2str(minSize(2)), '_mod16_2048/');

save_type = [1,0,1,0]; % img, density, dot, vis

mkdir(output_path);

path_img = strcat(output_path, '/img/'); mkdir(path_img);
path_den = strcat(output_path, '/den/');mkdir(path_den);
path_dot = strcat(output_path, '/dot/');mkdir(path_dot);
path_vis = strcat(output_path, '/vis/');mkdir(path_vis);

img_list = dir(fullfile(strcat(path,'/images/'),'*jpg')); 


for i_img = 1:size(img_list,1)
    img_name = img_list(i_img).name;
    name_split = regexp(img_name, '.jpg', 'split');       
    mat_name = strcat(name_split{1}, '.mat');

    im = imread(strcat(path, '/images/', img_name));
    [h, w, c] = size(im);

    %% resize and save img
    rate = init_maxSize(1)/h;
    rate_w = w*rate;
    if rate_w>init_maxSize(2)
        rate = init_maxSize(2)/w;
    end
    tmp_h = int16(h*rate/16)*16;

    if tmp_h < minSize(1)
        rate = minSize(1)/h;
    end
    tmp_w = int16(w*rate/16)*16;

    if tmp_w < minSize(2)
        rate = minSize(2)/w;
    end
    tmp_h = int16(h*rate/16)*16;
    tmp_w = int16(w*rate/16)*16;

    rate_h = double(tmp_h)/h;
    rate_w = double(tmp_w)/w;
    im = imresize(im,[tmp_h,tmp_w]);

    if save_type(1) == 1
        imwrite(im, strcat(path_img, img_name));
    end
    
    if ~exist(strcat(path, '/mats/',mat_name),'file')
        continue;
    end
    
    %% load mat 
    load(strcat(path, '/mats/',mat_name));
    
    if ~isempty(annPoints)
        annPoints(:,1) = annPoints(:,1)*double(rate_w);
        annPoints(:,2) = annPoints(:,2)*double(rate_h);   

        check_list = ones(size(annPoints,1),1);
        for j = 1:size(annPoints,1)
            x = ceil(annPoints(j,1)); 
            y = ceil(annPoints(j,2));
            if(x > tmp_w || y > tmp_h || x<=0 || y<=0)
                check_list(j) = 0;
            end
        end

        annPoints = annPoints(logical(check_list),:);
    end
    %% gen & save labels
    if save_type(2) == 1
        %% density generation   
        im_density = get_density_map_gaussian(im,annPoints,15,4); 
        csvwrite(strcat(path_den, name_split{1}, '.csv'), im_density);       
    end

    if save_type(3) ==1
        %% dot generation
        im_dots = get_dot_map(im,annPoints);
        im_dots = uint8(im_dots);
        imwrite(im_dots, strcat(path_dot, name_split{1}, '.png'));
    end
    
    if save_type(4) == 1
        %% visualization 
        if ~isempty(annPoints)
            imRGB = insertShape(im,'FilledCircle',[annPoints(:,1),annPoints(:,2),5*ones(size(annPoints(:,1)))],'Color', {'red'});
        else
            imRGB = im;
        end
        imwrite(imRGB, strcat(path_vis, name_split{1}, '.jpg'));   
    end 
        
end
