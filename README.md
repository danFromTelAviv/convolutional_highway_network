# convolutional_highway_network
In this file I added two layers ( keras - tensorflow ) that can be used in the following manner:
layer_1 = half_and_half_layer(layer_0, args**)

Note that this is not like the typical functional layers in which you end the line with "(layer_0)"

The two layer implemented are :
1) a convolutional highway layer
2) a convolutional glu layer that is contrained to passing exactly half of the information onwards. 

In practice the half_and_half_layer ( glu layer ) is as effective as highway layer and works much much faster. 

Cheers,
Dan


