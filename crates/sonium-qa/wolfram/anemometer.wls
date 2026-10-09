(* Literal numerical extractor for Wolfram's MEMS thermal Anemometer example. *)
Needs["NDSolve`FEM`"];

vars = {u[x, y], {x, y}};
scale = QuantityMagnitude[UnitConvert[Quantity[1, "Millimeters"], "Meters"]];
baseX = baseY = 0;
anemometerLength = 1200 scale;
additionalModelLength = 0;
anemometerHight = 80 scale;
anemometerBase = 240 scale;
membraneThick = 20 scale;
heatLength = 20 scale;
heatHight = 10 scale;
flowHight = 300 scale;

alpha = ArcTan[Sqrt[2]];
myBeta = Pi/2 - alpha;
membraneHight = anemometerHight - membraneThick;
l1 = membraneHight/Sin[alpha] Sin[myBeta];

p1 = {baseX, baseY};
p2 = p1 + {anemometerBase, 0};
p5 = p2 + {l1, membraneHight};
p4 = p1 + {anemometerLength, 0};
p3 = p4 - {anemometerBase, 0};
p6 = p3 - {l1, -membraneHight};
p9 = p1 + {0, anemometerHight};
p12 = p4 + {0, anemometerHight};
heatBase = (p12 - p9)/2 - {heatLength/2, -(anemometerHight - heatHight)};
p7 = heatBase;
p8 = heatBase + {heatLength, 0};
p10 = heatBase + {0, heatHight};
p11 = heatBase + {heatLength, heatHight};
p13 = p1 + {0, anemometerHight + flowHight};
p14 = p13 + {anemometerLength + additionalModelLength, 0};
coords = {p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14};

siliconIncidents = {{1, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 12},
  {12, 11}, {11, 10}, {10, 9}, {9, 1}};
heaterIncidents = {{10, 7}, {7, 8}, {8, 11}};
flowBoxIncidents = {{12, 14}, {14, 13}, {13, 9}};
freeCutIncidents = {{2, 3}};
deviceIncidents = Join[siliconIncidents, heaterIncidents, flowBoxIncidents, freeCutIncidents];
siliconMarker = {1, 7, 7, 7, 1, 7, 7, 7, 7, 7};
heaterMarker = {7, 7, 7};
flowBoxMarker = {2, 4, 3};
freeCutMarker = {1};
deviceMarker = Join[siliconMarker, heaterMarker, flowBoxMarker, freeCutMarker];
pointIncidents = Partition[Range[Length[coords]], 1];
pointMarkers = {1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4};
bmesh = ToBoundaryMesh[
  "Coordinates" -> coords,
  "BoundaryElements" -> {LineElement[deviceIncidents, deviceMarker]},
  "PointElements" -> {PointElement[pointIncidents, pointMarkers]}
];

p17 = p1 + {anemometerLength/2, (anemometerHight - membraneThick)/2};
p18 = p1 + {anemometerLength/2,
  anemometerHight - heatHight - (membraneThick - heatHight)/2};
p19 = p1 + {anemometerLength/2, anemometerHight - heatHight/2};
p20 = p1 + {anemometerLength/2, anemometerHight + flowHight/2};
mesh = ToElementMesh[bmesh,
  "RegionMarker" -> Transpose[{
    {p17, p18, p19, p20},
    {3, 1, 2, 4},
    {200., 200., 0.1, 200.} scale^2
  }]
];

kappa = Quantity[
  If[ElementMarker == 1 || ElementMarker == 2, 148, 0.0239],
  "Watts"/("Kelvins" "Meters")
];
velocity = 10^-3;
flowField = Quantity[{
  If[ElementMarker == 4,
    Evaluate[-velocity (((y - (anemometerHight + flowHight/2))/(flowHight/2))^2 - 1.)],
    0.],
  0.
}, "Meters"/"Seconds"];
source = Quantity[If[ElementMarker == 2, 10^7, 0], "Watts"/"Meters"^3];
md = Quantity[
  If[ElementMarker == 1 || ElementMarker == 2, 2330, 1.040],
  "Kilograms"/"Meters"^3
];
shc = Quantity[
  If[ElementMarker == 1 || ElementMarker == 2, 710, 1012],
  "Joules"/("Kilograms" "Kelvins")
];
pars = <|
  "ThermalConductivity" -> kappa,
  "HeatConvectionVelocity" -> flowField,
  "HeatSource" -> source,
  "MassDensity" -> md,
  "SpecificHeatCapacity" -> shc
|>;
op = HeatTransferPDEComponent[vars, pars];
bcs = HeatTemperatureCondition[ElementMarker == 1, vars, pars];
ifun = NDSolveValue[{op == 0, bcs}, u, Element[{x, y}, mesh]];

temperatureRange = MinMax[ifun["ValuesOnGrid"]];
sensorDifference = ifun[800 scale, 80 scale] - ifun[400 scale, 80 scale];
payload = <|
  "schema_version" -> 1,
  "maximum_temperature_k" -> N[Last[temperatureRange], 16],
  "sensor_difference_k" -> N[sensorDifference, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
