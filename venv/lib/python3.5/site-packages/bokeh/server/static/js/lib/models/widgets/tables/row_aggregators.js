"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var slickgrid_1 = require("slickgrid");
var _a = slickgrid_1.Data.Aggregators, Avg = _a.Avg, Min = _a.Min, Max = _a.Max, Sum = _a.Sum;
var p = require("../../../core/properties");
var model_1 = require("../../../model");
var RowAggregator = /** @class */ (function (_super) {
    tslib_1.__extends(RowAggregator, _super);
    function RowAggregator(attrs) {
        return _super.call(this, attrs) || this;
    }
    RowAggregator.initClass = function () {
        this.prototype.type = 'RowAggregator';
        this.define({
            field_: [p.String, ''],
        });
    };
    RowAggregator.__name__ = "RowAggregator";
    return RowAggregator;
}(model_1.Model));
exports.RowAggregator = RowAggregator;
RowAggregator.initClass();
var avg = new Avg();
var AvgAggregator = /** @class */ (function (_super) {
    tslib_1.__extends(AvgAggregator, _super);
    function AvgAggregator() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.key = 'avg';
        _this.init = avg.init;
        _this.accumulate = avg.accumulate;
        _this.storeResult = avg.storeResult;
        return _this;
    }
    AvgAggregator.initClass = function () {
        this.prototype.type = 'AvgAggregator';
    };
    AvgAggregator.__name__ = "AvgAggregator";
    return AvgAggregator;
}(RowAggregator));
exports.AvgAggregator = AvgAggregator;
AvgAggregator.initClass();
var min = new Min();
var MinAggregator = /** @class */ (function (_super) {
    tslib_1.__extends(MinAggregator, _super);
    function MinAggregator() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.key = 'min';
        _this.init = min.init;
        _this.accumulate = min.accumulate;
        _this.storeResult = min.storeResult;
        return _this;
    }
    MinAggregator.initClass = function () {
        this.prototype.type = 'MinAggregator';
    };
    MinAggregator.__name__ = "MinAggregator";
    return MinAggregator;
}(RowAggregator));
exports.MinAggregator = MinAggregator;
MinAggregator.initClass();
var max = new Max();
var MaxAggregator = /** @class */ (function (_super) {
    tslib_1.__extends(MaxAggregator, _super);
    function MaxAggregator() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.key = 'max';
        _this.init = max.init;
        _this.accumulate = max.accumulate;
        _this.storeResult = max.storeResult;
        return _this;
    }
    MaxAggregator.initClass = function () {
        this.prototype.type = 'MaxAggregator';
    };
    MaxAggregator.__name__ = "MaxAggregator";
    return MaxAggregator;
}(RowAggregator));
exports.MaxAggregator = MaxAggregator;
MaxAggregator.initClass();
var sum = new Sum();
var SumAggregator = /** @class */ (function (_super) {
    tslib_1.__extends(SumAggregator, _super);
    function SumAggregator() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.key = 'sum';
        _this.init = sum.init;
        _this.accumulate = sum.accumulate;
        _this.storeResult = sum.storeResult;
        return _this;
    }
    SumAggregator.initClass = function () {
        this.prototype.type = 'SumAggregator';
    };
    SumAggregator.__name__ = "SumAggregator";
    return SumAggregator;
}(RowAggregator));
exports.SumAggregator = SumAggregator;
SumAggregator.initClass();
