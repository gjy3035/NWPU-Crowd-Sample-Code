function dot_map = get_dot_map(im,points)

dot_map = zeros(size(im,1),size(im,2));
[h,w] = size(dot_map);

if(length(points)==0)
    return;
end

points = int32(points);
for i_dot = 1:size(points,1)
    
    x = points(i_dot,1);
    y = points(i_dot,2);
    
    if(x > w || y > h || x<=0 || y<=0)
        continue;
    end
    
    dot_map(y,x) = dot_map(y,x) + 1;
end

end


