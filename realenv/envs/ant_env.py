from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.envs.env_bases import *
from realenv.core.physics.robot_locomotors import Ant, AntClimber
from transforms3d import quaternions
from realenv import configs
import numpy as np
import sys
import pybullet as p
from realenv.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data

ANT_SENSOR_RESULT = [[-0.564215040895303, -1.2085311307338866, 1.3402042140736032],
[-0.40455374388885995, -1.6119284432360832, 1.4174799333736654],
[-0.824344002820735, -1.5500635155747196, 1.1493331682781847],
[-1.0578541283343885, -1.3847006500314791, 1.2035215067589935],
[-0.9408908549452564, -1.3387942696847523, 1.2673585454596425],
[-1.9680449166643732, -1.3054710626063504, 0.4381931332892494],
[-0.5844961152192064, -1.4756950247932599, 1.4876118305394794],
[-1.980738289853393, -1.4228512732220397, 0.284876104054924],
[-1.8626815368550786, -1.5455050913171302, 0.3729489442483859],
[-0.9361795389339641, -1.6731891045518918, 1.2286832032177373],
[-1.0007213426761772, -1.2606000442339325, 1.1250579287609375],
[-0.7784351600590133, -1.339761740992188, 1.140822500055486],
[-0.31243044075832144, -1.4643861007008778, 1.4234178756762992],
[-1.250996907035185, -1.265651587594828, 1.006943871306271],
[-1.9883183752499898, -1.7015350051123816, 0.389687929824162],
[-2.5264225925433434, -1.6460615661659685, 0.13142904641726036],
[-2.0869546291943406, -1.4517686715159481, 0.3669869527896669],
[-0.9709297413410478, -1.6305823072797159, 1.1994732996967306],
[-1.5257852471051094, -1.4822192123093425, 0.6870807395912478],
[-1.2000449724555173, -1.4669467537280554, 0.9650901502727457],
[-1.348546014487674, -1.5571243317296466, 0.8288859308321186],
[-0.7347313719438869, -1.15841153722043, 1.3545435322886608],
[-1.776731195249828, -1.3832073699365848, 0.6084095085997845],
[-0.8958899068839352, -1.3315204215645593, 1.1997851752480724],
[-1.1787713341038204, -1.4786988977433235, 0.7978761745017355],
[-1.0391371034456702, -1.3461039103328734, 1.0189594108246904],
[-2.1407113989790294, -1.6532717918389104, 0.31703614322298274],
[-1.7011283271083988, -1.6459719503653858, 0.6360052564450271],
[-2.473852507792975, -1.321304234019933, 0.2943985871766358],
[-1.2364227476850636, -1.5540574718969284, 0.7567679915363901],
[-1.98960096093765, -1.5091217090010434, 0.34546708612058885],
[-1.0535149282587084, -1.4961762922914712, 1.1365470929968307],
[-2.2218946275301352, -0.4167784057143221, 0.3297015824694125],
[-2.197653849911695, -1.260008136044545, 0.32609023176546226],
[-1.7265837417183807, -1.764640342316113, 0.6836949625855147],
[-0.8588718705750007, -1.3440780668466326, 1.2895569167018757],
[-0.5906770467937978, -1.6017868496557983, 1.3221370060799398],
[-1.0818365190910801, -1.6688009473709504, 0.9536549224315152],
[-1.1202604369883915, -1.4005066748709425, 0.9416879003873071],
[-2.1217692500371577, -1.3630650445091295, 0.36782074698241723],
[-1.6894536293977593, -1.5457088545739335, 0.6078618098375721],
[-1.9649936974100877, -1.261004976253715, 0.419217745207202],
[-1.0438205829077476, -1.5618009042624392, 1.2026393307415708],
[-0.3551761108714242, -1.6862917556782988, 1.4691583316558636],
[-0.2917501398523971, -1.2509898545310507, 1.6101155654440338],
[-1.5042453460065062, -1.4434242843662002, 0.5981129423903423],
[-0.9986407802246219, -1.2727669181653676, 1.1682351776743258],
[-2.2911994959337854, -1.5212959377333066, 0.25712663042976125],
[-0.5314961609799862, 0.4997171702734739, 0.24286497416497316],
[-0.8357970151222267, -1.6223335971753643, 1.3427632606063662],
[-1.1245175511036247, -1.3427455903212115, 1.0357435694621109],
[-2.214697546983776, -1.4302910192561227, 0.36653840192014386],
[-1.1202152804820995, -1.6677824814294708, 1.086344309028606],
[-2.392278243125399, -2.657546642604811, 0.2261882998373346],
[-0.8528048062002259, -1.7713896938773577, 1.2692720478789488],
[-0.8713102703968536, -1.4075718680689837, 1.1907741250390051],
[-0.5719443894848076, -1.4804238581927769, 1.4535934643996635],
[-1.804328335734409, -1.4013068295713602, 0.5226928856722844],
[-2.105199551916654, -1.7553642941212937, 0.2983588191059714],
[-1.9939710017873167, -1.5471938601180195, 0.351859812239505],
[-1.7655273847195334, -1.3437340005871232, 0.607669983900014],
[-0.9306020463078797, -1.6859952245530498, 1.1033986102714837],
[-1.3188354867014775, -1.4632904561200137, 0.933069670045651],
[-1.2990839176480855, -1.2264133339185916, 0.870038883731682],
[-0.820520930260256, -1.303066518501505, 1.2823385098940447],
[-0.871492566836802, -1.5247279074240248, 1.280499832987296],
[-2.1957165359858877, -1.7708736992269711, 0.357998027705012],
[-1.5457698706166283, -1.4387546743263868, 0.6702103657766793],
[-1.1839217858551807, -1.558048681047011, 0.9438497203359871],
[-1.2546237877838735, -1.402996452252915, 0.8602396663172308],
[-2.49281841122012, -0.5830402390765279, 0.20951058068719536],
[-1.2257507879147476, -1.4015615561665722, 1.0052237259588026],
[-1.5729417951514908, -1.6032621453715727, 0.7574318372834692],
[-0.8997263912546438, -1.2882771488735087, 1.1199815227180112],
[0.14155559251569638, -1.5250067324734489, 1.60154606538197],
[-1.5052353057074144, -1.6694243026092679, 0.7312637033608657],
[-1.1816553192473291, -1.5364744276707636, 0.9104084810827772],
[-0.8646339185180756, -1.4069445114126633, 1.2525024832386287],
[-0.8745043715938443, -1.5757978027625756, 1.2008900074261302],
[-1.8612378435432617, -1.428516973195013, 0.5683424196120603],
[-1.6610058359909297, -1.5023596382746955, 0.6856438025769956],
[-1.6367812085174014, -1.6383257021332298, 0.6326862411977342],
[-2.304851175899416, -0.9371657798316406, 0.24347656136524987],
[-0.8477489165970928, -1.653838267437248, 1.2826900140840183],
[-1.0734684438220794, -1.3805886287464726, 0.9652810200976876],
[-0.7328527953634232, -1.7019099252376193, 1.427270765401642],
[-1.373088353287598, -1.578047417935322, 0.8498122574671585],
[-1.976741420442828, -1.5335188085322076, 0.3835752585563181],
[-0.7145490223978649, -1.5475100974523623, 1.3551707736501746],
[-1.1412988581844281, -1.4836644892887019, 1.0474965359777317],
[-0.944498053282235, -1.267526854736348, 1.1344798270008574],
[-0.6676226393514019, -1.4266652488915081, 1.4273467384637972],
[-1.5770152284243486, -1.499404226461007, 0.7265957549203562],
[-1.0723947592401701, -1.5515955569812436, 1.1242548178681375],
[-1.2276717709776792, -1.3474369843815075, 0.9521695181162894],
[-1.1379082815599668, -1.4825047300359335, 0.9746890169564831],
[-2.085887166728974, 1.6100334055348016, 0.2300058687540219],
[-1.0149548123116685, -1.4674146825953458, 1.0772362056743194],
[-2.4181018316239213, -1.572861004735216, 0.21800130414864088],
[-2.414629053327871, -2.662122333007115, 0.2196250416015913],
[-1.1474948105045653, -1.5851811494580337, 1.0253557716783337],
[-0.9136501940408691, -1.4065812467279055, 1.178632411660507],
[-1.4083139866426069, -1.687877067360579, 0.9745828393804215],
[-1.957025546335101, -1.105521168427332, 0.1383859929183185],
[-2.192177863804532, -1.9486593345624932, 0.3838194755510696],
[-0.8253559417027437, -1.3476541631282946, 1.0896382517775638],
[-1.4590648225964673, -1.4110234082395066, 0.7810664065857871],
[-0.9051723658230193, -1.5050911777063436, 1.1905863002482893],
[-0.9641436771649663, -1.280040717017064, 1.1122200464190324],
[-1.327170481676561, -1.4450800507271613, 0.7895522995481544],
[-1.7643042268143956, -1.5098620551580202, 0.5116275866149117],
[-2.2061329800212675, -1.0093638072569446, 0.22634125077631273],
[-0.8348167671749284, -1.4369867515299624, 1.1975422896982226],
[-0.8604657326684265, -1.4970580468974475, 1.082283562179084],
[-1.1083282670110428, -1.6509948208995728, 1.0321560591709291],
[-2.0506736132215155, -1.8455088209884463, 0.36057412185986676],]


ANT_DEPTH_RESULT = [
[-0.681633074643924, -1.5929431066248143, 1.2892505926395408],
[-1.5242105636830268, -1.6926538220697886, 0.6676844568795478],
[-0.6569380453545255, -1.4339922062003614, 1.2609683373521188],
[-1.1577933160630323, -1.49872735482112, 0.9776951376483247],
[-0.9277320864261319, -1.4728562580188187, 1.0578041423083853],
[-1.4802932740174197, -1.7697428289893045, 0.6560647554139578],
[-1.3528014270244522, -1.40745657223757, 0.8717363331503329],
[-1.1482057970780601, -1.4774406923214911, 0.9196625196676794],
[-0.7718092947557101, -1.6762355976553203, 1.3460127574202934],
[-0.969181367291203, -1.4330100809820692, 1.0748351134798977],
[-2.1692948109567673, -1.5005682522593597, 0.17606267109211612],
[-2.247931046929969, -1.3872994829695924, 0.12252065600250292],
[-0.6741660974889446, -1.715365655317196, 1.3124876616987387],
[-0.9874152089003304, -1.569740482063952, 1.0906884175568605],
[-1.2276107973279395, -1.404879379914308, 1.0576261355132968],
[-2.008466532534012, -1.3373972135254808, 0.27018599464790566],
[-0.8527874125681424, -1.6899604548292055, 1.1985375622445642],
[-1.1791574032215002, -1.336764423171048, 0.9279028639873392],
[-0.6632702580762656, -1.5332324327932128, 1.2870425240623131],
[-1.4419824202272213, -1.6931000102236515, 0.7217499659927699],
[-1.1867608970168637, -1.404874271634229, 0.9059103579372435],
[-1.9803956691189348, -1.4590057663256475, 0.484339579927086],
[-0.7920263075648507, -1.1614570144007863, 1.277977702913847],
[-1.2290275914916173, -1.6504294025219768, 0.9326091101029083],
[-1.1680122277459952, -1.6016535698559495, 0.8363310038745122],
[-2.1873901080486426, -1.2630073351566846, 0.203868971390579],
[-0.9413326142524587, -1.338171728901452, 1.139952951698678],
[-1.0108287745385096, -1.5356924203917268, 1.1088757455642717],
[-0.9419440152216777, -1.7061315796556134, 1.0955595097036834],
[-2.1750326220778855, -2.7274478374785995, 0.24416095746847535],
[-1.1078808578624515, -1.6248364655936478, 0.9622277722305277],
[-0.29136388563507243, -1.5899196092992163, 1.5046267709303713],
[-0.9228290326098718, -1.4679313363751085, 0.9903483143719952],
[-0.7859733902357892, -1.5911150953282631, 1.278266038849065],
[-0.9426700895830631, -1.7267655377137308, 1.083398148664682],
[-0.8095351578699158, -1.7261235489872622, 1.3133696226716236],
[-1.4687327126542096, -1.5303563998373704, 0.8434179136350803],
[-0.9583676025058093, -1.4018138176443924, 1.0891987881359837],
[-2.0292200852059508, -1.2942425413545688, 0.2896303285491902],
[-1.039142687339294, -1.6367987449167782, 0.8928516827249005],
[-1.3229869343628162, -1.336229541353571, 0.937259310780825],
[-0.6998706140615896, -1.5002424831893155, 1.260700754680594],
[-1.1837996956219816, -1.6315386075608211, 0.9526131280622219],
[-0.45970774218653154, -1.6353582525761399, 1.3965814142248314],
[-0.9393722134903352, -1.3913177730887774, 1.1374892870161066],
[-1.2322775831161248, -1.498758224404743, 0.9822500011472312],
[-1.2189148801267518, -1.7229743940623792, 0.9048385279493427],
[-2.39652837263437, -1.8498078893447474, 0.33792085556402374],
[-0.9364744585162221, -1.5306037323792794, 1.123920041991977],
[-0.8900177412340277, -1.3975568825337459, 1.0986131033604587],
[-0.8976293331476025, -1.4697975460408523, 1.179635273175759],
[-0.9752649695097836, -1.556822513204339, 1.238471493744246],
[-0.44384442051056394, -1.6594928375326998, 1.425115401277467],
[-0.9137578439623787, -1.3347119347962466, 1.20486385152254],
[-1.321388608192472, -1.7140727859328988, 0.8111669393752965],
[-1.1791829444833974, -1.5914558776620353, 0.9710091016151112],
[-1.0268640799906716, -1.2705610173954098, 1.0760151447602928],
[-1.3246515669000523, -1.4532439721807957, 0.8042669205412424],
[-0.9990013816971886, -1.4171253184393442, 1.077702760936969],
[-0.9693643391744485, -1.6582222046968689, 1.063373880695584],
[-0.7645287060780143, -1.6102672602621664, 1.2636699546007384],
[-0.7451288635063701, -1.639101732206342, 1.2329547571108899],
[-1.6159999760334625, -1.6247071736358765, 0.5499195908155731],
[-0.7740981395436747, -1.3757638418717946, 1.2587040056140995],
[-1.3885286389590128, -1.5585869633115765, 0.7755943263996363],
[-1.027259375602486, -1.5539332843336104, 0.9324676246360217],
[-1.2078141270580023, -1.4182419490546079, 0.7393503176471563],
[-2.0078859873841637, -1.5915727996851445, 0.2964686295863236],
[-0.7675689000092328, -1.6308321033975053, 1.2651777306372212],
[-1.3921561884340048, -1.568220831800571, 0.8678415450015408],
[-2.155762560107032, -1.61726169891486, 0.42875127875182545],
[-1.178244409544884, -1.5130490454537298, 0.9895983180891776],
[-1.0097713077937092, -1.5733789243156797, 0.9761249968903488],
[-1.5343438189709127, -1.3998886911885664, 0.7518528242734657],
[-0.9724951868289616, -1.4044962191703279, 1.2407648661354693],
[-0.9215055329625785, -1.377721291858359, 1.2577585548931554],
[-0.4589532474951384, -1.5225852846060077, 1.420555319855956],
[-0.6586809433347177, -1.4917272970032174, 1.3195231880142229],
[-0.9293783562552703, -1.411396933113772, 1.2226683421394906],
[-0.7951995851510013, -1.3955435663152305, 1.2286403622476834],
[-1.448187686523082, -1.276570700202863, 0.7474206473807827],
[-1.1083781055525543, -1.484920106280359, 0.9400631668066706],
[-0.8005766897529254, -1.658995288569123, 1.2399613581449689],
[-1.2830980395590281, -1.7336817284493622, 0.8333404596175137],
[-0.7124825561499877, -1.7018945598434922, 1.2985719714295496],
[-1.3988614689970167, -1.4612922999884093, 0.7511035465798711],
[-1.5985682899157154, -1.5584456074431188, 0.515102909899045],
[-0.6632540291693632, -1.7438859768845318, 1.3338349634850586],
[-1.7532363091075447, -1.6445007279493964, 0.5400349401602742],
[-0.6974583431088108, -1.5784127532182333, 1.2716898340903737],
[-1.0674329023114588, -1.3042403873351351, 0.9433688282026843],
[-0.8053176920277284, -1.40776653898824, 1.3358639538334423],
[-1.279132916578218, -1.62993339893065, 0.85065538345638],
[-1.5400857320266563, -1.3859810772352659, 0.758793564299635],
[-1.440623045061207, -1.4639191820128103, 0.7571199102451205],
[-1.2235000032138181, -1.5362267552865567, 1.0245041533009325],
[-0.88174021052127, -1.5720995481006077, 1.048247000342001],
[-0.9082613055283462, -1.7335234825619386, 1.5322684093452181],
[-1.9311357544708057, -1.2013512912714153, 0.5037967450332453],
[-1.7075472052406881, -1.4929559934517318, 0.5100590358451451],
[-1.2132433106052676, -1.356641469069574, 0.8717298879385633],
[-0.9424436238681244, -1.4331653529956323, 1.0898772413106468],
[-1.0773612413720253, -1.4687420426554814, 1.0365429794930254],
[-1.0328508397779552, -1.5706756266104747, 0.9778840503757319],
[-0.9903480439121737, -1.5905470084764084, 1.1333271585163036],
[-0.9312342228493271, -1.0827507758392363, 1.121614991544965],
[-1.0797126146502314, -1.6041829044034666, 1.057179356609098],
[-0.8096313234555421, -1.3345319051702182, 1.3415901127868644],
[-1.6588509754507337, -1.4144949191196157, 0.5412815864019013],
[-1.0205401266814162, -1.3641313987770902, 1.1593512006282836],
[-0.4301609271086237, -1.418257146821862, 1.4800998601426008],
[-0.9310206273895375, -1.576659794554351, 1.0532679262288618],
[-0.7502915003407026, -1.4129944666351202, 1.3266246606068206],
[-0.3700686139601712, -1.4010212651304883, 1.4155203137061998],
[-1.5405206187570064, -1.421009567417917, 0.7348209788135635],
[-0.7381059349049697, -1.6015328474156405, 1.1415718181785455],
[-1.0514608728966817, -1.5585585232897257, 1.1023195937096066],
[-0.4734916963668789, -1.7763690187590484, 1.3829929878837381],
[-1.7840758468025302, -1.3489062877291047, 0.5028072324610731],
[-1.1235759093368884, -1.730926382840435, 0.8843437437349186],
[-2.080271357697461, -1.2886992288185486, 0.26212575512234176],
[-2.0154787804326486, -1.5524986909000764, 0.4606061730989126],
[-1.14143119265786, -1.4471891117934634, 0.9477398212786093],
[-1.9576905340555866, -1.5089424168746028, 0.4110331125250219],
[-1.8584438756910002, -1.6931552819940583, 0.47119667356458633],
[-1.1618444547290665, -1.5132771557858464, 0.9123730353699017],
[-0.786039371104779, -1.3202410883516251, 1.0768218933945004],
[-2.0029166077460543, -1.6244527555476416, 0.2991862436320669],
[-1.2508880160592384, -1.6810782155093347, 1.0268817000718264],
[-0.9071134077622461, -1.6724997400044836, 1.1083838297610946],
[-2.4224475642374568, -1.94621970954711, 0.3176335844078091],
[-2.1020643846421816, -1.6772857766531515, 0.2891504230683036],
[-1.805186081976453, -1.2703813416711283, 0.4280150740895572],
[-0.7620646284653393, -1.3435817272934922, 1.3422146785151499],
[-2.065565670501401, -1.687021247481736, 0.12508909797328222],
[-0.8803382377599347, -1.4532954825210462, 1.1153410564152877],
[-1.2048581549842465, -1.275875736339281, 0.9301446046552172],
[-1.6088053927877985, -1.5689500569664332, 0.5277998651031077],
[-0.9325931061379097, -1.5872591343027265, 1.1262359005411209],
[-0.9504610777685852, -1.4880223182352517, 1.0265497158075805],
[-1.4815643851205744, -1.7740235114198943, 0.7089375433731966],
[-0.6054859775577871, -1.6112737307343599, 1.272656035938342],
[-0.47729374168121763, -1.652355379324228, 1.3984991440542165],
[-0.8390390793864582, -1.2359279856453438, 1.2867292803637371],
[-1.3571592079527213, -1.3940213176924692, 0.8643478318134951],
[-0.6835464125896749, -1.4893505334558712, 1.3854799283669086],
[-2.060980982360715, -1.3130027365224521, 0.3874706560008267],
[-0.6612894895562352, -1.4264688606930342, 1.3425242192043507],
[-0.934560213118626, -1.4409460740567466, 1.1353229833204033],
[-1.202959078659831, -1.2784037048501313, 0.8926402590952024],
[-0.5926035593029095, -1.4360473565369025, 1.23690613852692],
[-0.6138951947431677, -1.6371303167046642, 1.2869839416235385],
[-1.0107551365356182, -1.5531749788004712, 1.0238703477549187],
[-1.083619914615622, -1.5016621363660902, 0.9686197730260354],
[-0.8951622234283126, -1.5519056622747895, 1.094494688959465],
[-2.070095005799851, -1.4656868945246628, 0.48249463535967885],
[-0.8509151843804255, -1.2978805992358502, 1.2443385470963075],
[-0.7822471314311975, -1.4025613992877473, 1.2214767862946467],
[-1.234641606663068, -1.5367608774871742, 0.8882559094602271],

]

ANT_TIMESTEP  = 1.0/(4 * 22)
ANT_FRAMESKIP = 4

"""Task specific classes for Ant Environment
Each class specifies: 
    (1) Target position
    (2) Reward function
    (3) Done condition
    (4) (Optional) Curriculum learning condition
"""

tracking_camera = {
    'pitch': -20,
    # 'pitch': -24  # demo: stairs
    #self.tracking_camera['pitch'] = -45 ## stairs
    'yaw': 46,     ## demo: living room
    #yaw = 30    ## demo: kitchen
    'z_offset': 0.5,
    'distance': 5 ## living room
    #self.tracking_camera['yaw'] = 90     ## demo: stairs
}

class AntNavigateEnv(CameraRobotEnv):
    def __init__(
            self, 
            human=True, 
            timestep=ANT_TIMESTEP, 
            frame_skip=ANT_FRAMESKIP, 
            is_discrete=False, 
            mode="RGBD", 
            use_filler=True, 
            gpu_count=0, 
            resolution="NORMAL"):
        self.human = human
        self.model_id = configs.NAVIGATE_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos   = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][-1]
        initial_orn, initial_pos = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][0]
        
        self.robot = Ant(initial_pos, initial_orn, 
            is_discrete=is_discrete, 
            target_pos=target_pos,
            resolution=resolution)
        CameraRobotEnv.__init__(
            self, 
            mode, 
            gpu_count, 
            scene_type="building", 
            use_filler=use_filler)

        self.total_reward = 0
        self.total_frame = 0
        
        
    def calc_rewards_and_done(self, a, state):
        ### TODO (hzyjerry): this is directly taken from husky_env, needs to be tuned 

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()
       
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch

        done = self.nframe > 300
        #done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            # print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        # print(self.robot.feet_contact)

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we 
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        print("Frame %f reward %f" % (self.nframe, progress))
        return [
            #alive,
            progress,
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
         ], done

    def flag_reposition(self):
        walk_target_x = self.robot.walk_target_x / self.robot.mjcf_scaling
        walk_target_y = self.robot.walk_target_y / self.robot.mjcf_scaling

        self.flag = None
        if self.human and not configs.DISPLAY_UI:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x, walk_target_y, 0.5])
        
    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self.flag_reposition()
        return obs


class AntGoallessRunEnv(CameraRobotEnv):
    def __init__(
            self,
            human=True,
            timestep=ANT_TIMESTEP,
            frame_skip=ANT_FRAMESKIP,
            is_discrete=False,
            mode="RGBD",
            use_filler=True,
            gpu_count=0,
            resolution="NORMAL"):
        self.human = human
        self.model_id = configs.NAVIGATE_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][-1]
        initial_orn, initial_pos = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][0]

        self.robot = Ant(initial_pos, initial_orn,
                         is_discrete=is_discrete,
                         target_pos=target_pos,
                         resolution=resolution)
        CameraRobotEnv.__init__(
            self,
            mode,
            gpu_count,
            scene_type="building",
            use_filler=use_filler)

        self.total_reward = 0
        self.total_frame = 0

    def calc_rewards_and_done(self, a, state):
        ### TODO (hzyjerry): this is directly taken from husky_env, needs to be tuned

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch

        done = self.nframe > 300
        # done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_goalless_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            # print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        # print(self.robot.feet_contact)

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())  # let's assume we
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        print("Frame %f reward %f" % (self.nframe, progress))
        return [
                   # alive,
                   progress,
                   # electricity_cost,
                   # joints_at_limit_cost,
                   # feet_collision_cost
               ], done

    def flag_reposition(self):
        walk_target_x = self.robot.walk_target_x / self.robot.mjcf_scaling
        walk_target_y = self.robot.walk_target_y / self.robot.mjcf_scaling

        self.flag = None
        if self.human and not configs.DISPLAY_UI:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH,
                                                     fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                     meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1,
                                                 basePosition=[walk_target_x, walk_target_y, 0.5])

    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self.flag_reposition()
        return obs


class AntClimbEnv(CameraRobotEnv):
    delta_target = [configs.RANDOM_TARGET_RANGE, configs.RANDOM_TARGET_RANGE]

    def __init__(
            self, 
            human=True, 
            timestep=ANT_TIMESTEP, 
            frame_skip=ANT_FRAMESKIP,           ## increase frame skip, as the motion is too flashy 
            is_discrete=False, 
            mode="RGBD", 
            use_filler=True, 
            gpu_count=0, 
            resolution="NORMAL"):
        self.human = human
        self.model_id = configs.CLIMB_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos   = configs.TASK_POSE[configs.CLIMB_MODEL_ID]["climb"][-1]
        initial_orn, initial_pos = configs.TASK_POSE[configs.CLIMB_MODEL_ID]["climb"][0]

        self.target_pos_gt = target_pos
        self.robot = AntClimber(initial_pos, initial_orn, 
            is_discrete=is_discrete, 
            target_pos=target_pos,
            resolution=resolution,
            mode=mode)
        CameraRobotEnv.__init__(
            self, 
            mode, 
            gpu_count, 
            scene_type="building", 
            use_filler=use_filler)

        self.total_reward = 0
        self.total_frame = 0
        self.visual_flagId = None
        
        
    def calc_rewards_and_done(self, a, state):
        #time.sleep(0.1)
        ### TODO (hzyjerry): this is directly taken from husky_env, needs to be tuned 

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()
       
        alive = float(self.robot.alive_bonus(self.robot.body_rpy[0], self.robot.body_rpy[1]))  # state[0] is body height above ground (z - z initial), body_rpy[1] is pitch

        done = self.nframe > 700 or alive < 0 or self.robot.body_xyz[2] < 0

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            # print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        # print(self.robot.feet_contact)

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we 
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        
        close_to_goal = 0
        if self.robot.is_close_to_goal():
            close_to_goal = 1

        done = self.nframe > 600 or alive < 0 or close_to_goal

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True


        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        reward = [
            #alive,
            progress,
            #close_to_goal,
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
         ]

        debugmode = 0
        if (debugmode):
            print("reward")
            print(reward)
        return reward, done

    def randomize_target(self):
        if configs.RANDOM_TARGET_POSE:
            delta_x = self.np_random.uniform(low=-self.delta_target[0],
                                             high=+self.delta_target[0])
            delta_y = self.np_random.uniform(low=-self.delta_target[1],
                                             high=+self.delta_target[1])
        else:
            delta_x = 0
            delta_y = 0
        self.temp_target_x = (self.target_pos_gt[0] + delta_x)
        self.temp_target_y = (self.target_pos_gt[1] + delta_y)

    def flag_reposition(self):
        walk_target_x = self.temp_target_x
        walk_target_y = self.temp_target_y
        walk_target_z = self.robot.walk_target_z

        self.robot.walk_target_x = walk_target_x    # Change robot target accordingly
        self.robot.walk_target_y = walk_target_y    # Important for stair climbing
        self.robot.walk_target_z = walk_target_z

        self.flag = None
        if not self.human:
            return

        if self.visual_flagId is None:
            if configs.DISPLAY_UI:
                self.visual_flagId = -1
            else:
                self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x / self.robot.mjcf_scaling, walk_target_y / self.robot.mjcf_scaling, walk_target_z / self.robot.mjcf_scaling])

            
            for i in range(len(ANT_SENSOR_RESULT)):
                walk_target_x, walk_target_y, walk_target_z = ANT_SENSOR_RESULT[i]
                visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[0.5, 0, 0, 0.7])
                self.last_flagId = p.createMultiBody(baseVisualShapeIndex=visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x / self.robot.mjcf_scaling, walk_target_y / self.robot.mjcf_scaling, walk_target_z / self.robot.mjcf_scaling])
            
            for i in range(len(ANT_DEPTH_RESULT)):
                walk_target_x, walk_target_y, walk_target_z = ANT_DEPTH_RESULT[i]
                visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[0, 0.5, 0, 0.7])
                self.last_flagId = p.createMultiBody(baseVisualShapeIndex=visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x / self.robot.mjcf_scaling, walk_target_y / self.robot.mjcf_scaling, walk_target_z / self.robot.mjcf_scaling])
            
        else:
            last_flagPos, last_flagOrn = p.getBasePositionAndOrientation(self.last_flagId)
            p.resetBasePositionAndOrientation(self.last_flagId, [walk_target_x  / self.robot.mjcf_scaling, walk_target_y / self.robot.mjcf_scaling, walk_target_z / self.robot.mjcf_scaling], last_flagOrn)
        
    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        self.randomize_target()
        self.flag_reposition()
        obs = CameraRobotEnv._reset(self)       ## Important: must come after flat_reposition
        return obs




class AntFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=ANT_TIMESTEP,
                 frame_skip=ANT_FRAMESKIP, is_discrete=False, 
                 gpu_count=0):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        self.flag_timeout = 1
        self.tracking_camera = tracking_camera
        initial_pos, initial_orn = [0, 0, 1], [0, 0, 0, 1]
        self.robot = Ant(initial_pos, initial_orn, 
            is_discrete=is_discrete)
        CameraRobotEnv.__init__(
            self, 
            "SENSOR", 
            gpu_count, 
            scene_type="stadium", 
            use_filler=False)

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
        self.lastid = None


    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def flag_reposition(self):
        self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
                                                    high=+self.scene.stadium_halflen)
        self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
                                                    high=+self.scene.stadium_halfwidth)

        more_compact = 0.5  # set to 1.0 whole football field
        self.walk_target_x *= more_compact / self.robot.mjcf_scaling
        self.walk_target_y *= more_compact / self.robot.mjcf_scaling

        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 600 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.human:
            if self.lastid:
                p.removeBody(self.lastid)

            self.lastid = p.createMultiBody(baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=-1, basePosition=[self.walk_target_x, self.walk_target_y, 0.5])

        self.robot.walk_target_x = self.walk_target_x
        self.robot.walk_target_y = self.walk_target_y

    def calc_rewards_and_done(self, a, state):
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        head_touch_ground = 0
        if head_touch_ground == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = head_touch_ground > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(head_touch_ground)
            print("progress")
            print(progress)

        return [
            alive_score,
            progress,
        ], done


    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self.flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta


class AntFetchEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=ANT_TIMESTEP,
                 frame_skip=ANT_FRAMESKIP, is_discrete=False,
                 gpu_count=0, scene_type="building", mode = 'SENSOR'):

        target_orn, target_pos = INITIAL_POSE["ant"][configs.FETCH_MODEL_ID][-1]
        initial_orn, initial_pos = configs.INITIAL_POSE["ant"][configs.FETCH_MODEL_ID][0]

        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.model_id = configs.FETCH_MODEL_ID
        ## Mode initialized with mode=SENSOR
        self.tracking_camera = tracking_camera

        self.robot = Ant(
            is_discrete=is_discrete,
            initial_pos=initial_pos,
            initial_orn=initial_orn)

        CameraRobotEnv.__init__(
            self,
            mode,
            gpu_count,
            scene_type="building")
        self.flag_timeout = 1


        self.visualid = -1

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.5, 0.2])

        self.lastid = None
        
    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def flag_reposition(self):
        #self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        #self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)


        force_x = self.np_random.uniform(-300,300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        #self.walk_target_x *= more_compact
        #self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.body_xyz


        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 600 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.lastid:
            p.removeBody(self.lastid)

        self.lastid = p.createMultiBody(baseMass = 1, baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
        p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def calc_rewards_and_done(self, a, state):
        if self.lastid:
            ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)
            self.robot.walk_target_x = ball_xyz[0]
            self.robot.walk_target_y = ball_xyz[1]


        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        head_touch_ground = 1
        if head_touch_ground == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = head_touch_ground > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("head_touch_ground=")
            print(head_touch_ground)
            print("progress")
            print(progress)

        return [
            alive_score,
            progress,
        ], done


    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self.flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta


class AntFetchKernelizedRewardEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=ANT_TIMESTEP,
                 frame_skip=ANT_FRAMESKIP, is_discrete=False,
                 gpu_count=0, scene_type="building"):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        self.model_id = configs.FETCH_MODEL_ID

        self.flag_timeout = 1
        self.tracking_camera = tracking_camera

        initial_pos, initial_orn = configs.INITIAL_POSE["ant"][configs.FETCH_MODEL_ID][0]
        self.robot = Ant(initial_pos, initial_orn, 
            is_discrete=is_discrete)
        CameraRobotEnv.__init__(
            self, 
            "SENSOR", 
            gpu_count, 
            scene_type="building", 
            use_filler=False)


        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.5, 0.2])

        self.lastid = None
        
    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def flag_reposition(self):
        #self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        #self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)


        force_x = self.np_random.uniform(-300,300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        #self.walk_target_x *= more_compact
        #self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.body_xyz


        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 600 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.lastid:
            p.removeBody(self.lastid)

        self.lastid = p.createMultiBody(baseMass = 1, baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
        p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def calc_rewards_and_done(self, a, state):
        if self.lastid:
            ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)
            self.robot.walk_target_x = ball_xyz[0]
            self.robot.walk_target_y = ball_xyz[1]


        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        head_touch_ground = 1
        if head_touch_ground == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = head_touch_ground > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("head_touch_ground=")
            print(head_touch_ground)
            print("progress")
            print(progress)


        return [
            alive_score,
            progress,
        ], done


    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self.flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta
