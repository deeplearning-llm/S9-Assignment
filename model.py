import torch.nn as nn
import torch.nn.functional as F



dropout_value = 0.01

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1 input 32/1/1

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128, kernel_size=(3,3), padding=2, bias=False,stride=1),
         #   nn.Conv2d(in_channels=3,out_channels=32, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 32 , output_size = 34 , receptive_field = 3
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128,groups=128, out_channels=128, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 34 , output_size = 36 , receptive_field = 5
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=2, bias=False,stride=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
            #input_size = 36 , output_size = 18 , receptive_field =9
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, groups=32, out_channels=32, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=32,out_channels=256, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 18 , output_size = 20 , receptive_field = 13
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=256,groups=256, out_channels=256, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=256,out_channels=70, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 20 , output_size = 22 , receptive_field = 17
            nn.BatchNorm2d(70),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=70, out_channels=32, kernel_size=(3,3), padding=2, bias=False,stride=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
            #input_size = 22 , output_size = 11 , receptive_field =21
            )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, groups=32, out_channels=32, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 11 , output_size = 13 , receptive_field = 13
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, groups=128, out_channels=128, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 13 , output_size = 15 , receptive_field = 29
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3,3), padding=2, bias=False,stride=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)

           #input_size = 15 , output_size = 8 , receptive_field =33
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32,groups=32, out_channels=32, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=32,out_channels=100, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 8 , output_size = 10 , receptive_field = 49
            nn.BatchNorm2d(100),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=100,groups=100, out_channels=100, kernel_size=(3,3), padding=2, bias=False,stride=1),
            nn.Conv2d(in_channels=100,out_channels=64, kernel_size=(1,1), padding=0, bias=False,stride=1),
            nn.ReLU(), #input_size = 10 , output_size = 12 , receptive_field = 65
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3,3), padding=2, bias=False,stride=2),
            nn.BatchNorm2d(80)
            #input_size = 12 , output_size = 7 , receptive_field = 97
        )


        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) #input_size = 7 , output_size = 1 , receptive_field =129

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=10, kernel_size=(1,1), padding=0, bias=False,stride=1))



        #self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)