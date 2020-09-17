

class dictModify():
    def __init__(self, new_dict, old_dict):
        self.new_dict = new_dict
        self.old_dict = old_dict
        self.check_dict = ["context_encoding.stages.0.2.weight", "context_encoding.stages.0.2.bias", "context_encoding.stages.0.2.running_mean", "context_encoding.stages.0.2.running_var", "context_encoding.stages.1.2.weight", "context_encoding.stages.1.2.bias", "context_encoding.stages.1.2.running_mean", "context_encoding.stages.1.2.running_var", "context_encoding.stages.2.2.weight", "context_encoding.stages.2.2.bias", "context_encoding.stages.2.2.running_mean", "context_encoding.stages.2.2.running_var", "context_encoding.stages.3.2.weight", "context_encoding.stages.3.2.bias", "context_encoding.stages.3.2.running_mean", "context_encoding.stages.3.2.running_var", "context_encoding.bottleneck.1.weight", "context_encoding.bottleneck.1.bias", "context_encoding.bottleneck.1.running_mean", "context_encoding.bottleneck.1.running_var", "edge.conv1.1.weight", "edge.conv1.1.bias", "edge.conv1.1.running_mean", "edge.conv1.1.running_var", "edge.conv2.1.weight", "edge.conv2.1.bias", "edge.conv2.1.running_mean", "edge.conv2.1.running_var", "edge.conv3.1.weight", "edge.conv3.1.bias", "edge.conv3.1.running_mean", "edge.conv3.1.running_var", "decoder.conv1.1.weight", "decoder.conv1.1.bias", "decoder.conv1.1.running_mean", "decoder.conv1.1.running_var", "decoder.conv2.1.weight", "decoder.conv2.1.bias", "decoder.conv2.1.running_mean", "decoder.conv2.1.running_var", "decoder.conv3.1.weight", "decoder.conv3.1.bias", "decoder.conv3.1.running_mean", "decoder.conv3.1.running_var", "decoder.conv3.3.weight", "decoder.conv3.3.bias", "decoder.conv3.3.running_mean", "decoder.conv3.3.running_var", "fushion.1.weight", "fushion.1.bias", "fushion.1.running_mean", "fushion.1.running_var"]
        self.check_dict_2 =["context_encoding.stages.0.2.negative_slope.weight", "context_encoding.stages.0.2.negative_slope.bias", "context_encoding.stages.0.2.negative_slope.running_mean", "context_encoding.stages.0.2.negative_slope.running_var", "context_encoding.stages.1.2.negative_slope.weight", "context_encoding.stages.1.2.negative_slope.bias", "context_encoding.stages.1.2.negative_slope.running_mean", "context_encoding.stages.1.2.negative_slope.running_var", "context_encoding.stages.2.2.negative_slope.weight", "context_encoding.stages.2.2.negative_slope.bias", "context_encoding.stages.2.2.negative_slope.running_mean", "context_encoding.stages.2.2.negative_slope.running_var", "context_encoding.stages.3.2.negative_slope.weight", "context_encoding.stages.3.2.negative_slope.bias", "context_encoding.stages.3.2.negative_slope.running_mean", "context_encoding.stages.3.2.negative_slope.running_var"]
        self.check_dict_3 = ["context_encoding.bottleneck.1.weight", "context_encoding.bottleneck.1.bias", "context_encoding.bottleneck.1.running_mean", "context_encoding.bottleneck.1.running_var", "edge.conv1.1.weight", "edge.conv1.1.bias", "edge.conv1.1.running_mean", "edge.conv1.1.running_var", "edge.conv2.1.weight", "edge.conv2.1.bias", "edge.conv2.1.running_mean", "edge.conv2.1.running_var", "edge.conv3.1.weight", "edge.conv3.1.bias", "edge.conv3.1.running_mean", "edge.conv3.1.running_var", "decoder.conv1.1.weight", "decoder.conv1.1.bias", "decoder.conv1.1.running_mean", "decoder.conv1.1.running_var", "decoder.conv2.1.weight", "decoder.conv2.1.bias", "decoder.conv2.1.running_mean", "decoder.conv2.1.running_var", "decoder.conv3.1.weight", "decoder.conv3.1.bias", "decoder.conv3.1.running_mean", "decoder.conv3.1.running_var", "decoder.conv3.3.weight", "decoder.conv3.3.bias", "decoder.conv3.3.running_mean", "decoder.conv3.3.running_var", "fushion.1.weight", "fushion.1.bias", "fushion.1.running_mean", "fushion.1.running_var"]
        for k, v in self.old_dict.items():
            name = k[7:]  # remove `module.`
            name = self.find_match_2(inputs = name)
            self.new_dict[name] = v


    
    def find_match(self, inputs):
        """
        Example : given a name like 
            *context_encoding.stages.0.2.weight
        the output should find a coresponding name in the old state_dict
        in this scenario, the new name simply just 
            *context_encoding.stages.0.2.negative_slope.weight
        """

        parts = inputs.split(".")
        if inputs in self.check_dict:
            name = ""
            parts.insert(-1, "negative_slope")
            for i, part in enumerate(parts):
                name += ".{}".format(part) if i!=0 else part
            return name
        if "negative_slope" in self.check_dict_2:
            parts.remove("negative_slope")
            for i, part in enumerate(parts):
                name += ".{}".format(part) if i!=0 else part
            return name            
        return inputs

        # name = ""
        # for i, part in enumerate(parts):
        #     if part != "negative_slope":
        #         name += ".{}".format(part) if i!=0 else part
        #     else:
        #         continue
        # return name

    def find_match_2(self, inputs):
        """
        Example : given a name like 
            *context_encoding.stages.0.2.weight
        the output should find a coresponding name in the old state_dict
        in this scenario, the new name simply just 
            *context_encoding.stages.0.2.bn.weight
        """

        if inputs in self.check_dict_3:
            parts = inputs.split(".")
            name = ""
            parts.insert(-1, "bn")
            for i, part in enumerate(parts):
                name += ".{}".format(part) if i!=0 else part
            return name
        else:
            return inputs

    def arange(self):
        return self.new_dict