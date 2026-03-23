#include "tests.h"
#include "testtools.h"


#include "additiontests.h"
#include "subtractiontests.h"
#include "multiplicationtests.h"
#include "divisiontests.h"
#include "sigmoidtests.h"
#include "mselosstests.h"


void RunAdditionTests(Inferno::Device device);
void RunSubtractionTests(Inferno::Device device);
void RunMultiplicationTests(Inferno::Device device);
void RunDivisionTests(Inferno::Device device);
void RunSigmoidTests(Inferno::Device device);
void RunMSELossTests(Inferno::Device device);


void RunTests() {
    RunAdditionTests(Inferno::Device::cpu());
    RunAdditionTests(Inferno::Device::cuda(0));

    RunSubtractionTests(Inferno::Device::cpu());
    RunSubtractionTests(Inferno::Device::cuda(0));

    RunMultiplicationTests(Inferno::Device::cpu());
    RunMultiplicationTests(Inferno::Device::cuda(0));

    RunDivisionTests(Inferno::Device::cpu());
    RunDivisionTests(Inferno::Device::cuda(0));

    RunSigmoidTests(Inferno::Device::cpu());
    RunSigmoidTests(Inferno::Device::cuda(0));

    RunMSELossTests(Inferno::Device::cpu());
    RunMSELossTests(Inferno::Device::cuda(0));


}




		

