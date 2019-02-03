import Foundation
import CreateML

let training = URL(fileURLWithPath: "/tmp/ml/data-applenews/training")
let testing = URL(fileURLWithPath: "/tmp/ml/data-applenews/training")
let model = try MLImageClassifier(trainingData: .labeledDirectories(at: training))
let evaluation = model.evaluation(on: .labeledDirectories(at: testing))
try model.write(to: URL(fileURLWithPath: "/tmp/ml/data-applenews/applemews.mlmodel"))

