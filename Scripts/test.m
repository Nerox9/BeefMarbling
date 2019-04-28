function output = test(layer, input)
nLayers = 3;
    for j=1:1:nLayers
        if (j==1)
            layer(j).out = feval(layer(j).func,layer(j).weight*a0+layer(j).bias);
        elseif(j==nLayers)
            layer(j).out = feval(layer(j).func,layer(j).weight*layer(j-1).out+layer(j).bias);
        else
            layer(j).out = feval(layer(j).func,layer(j).weight*layer(j-1).out+layer(j).bias);
        end
    end
    output=layer(nLayers).out;
end