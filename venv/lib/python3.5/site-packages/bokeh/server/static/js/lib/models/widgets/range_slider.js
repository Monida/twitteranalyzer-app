"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var numbro = require("numbro");
var abstract_slider_1 = require("./abstract_slider");
var RangeSliderView = /** @class */ (function (_super) {
    tslib_1.__extends(RangeSliderView, _super);
    function RangeSliderView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    RangeSliderView.__name__ = "RangeSliderView";
    return RangeSliderView;
}(abstract_slider_1.AbstractRangeSliderView));
exports.RangeSliderView = RangeSliderView;
var RangeSlider = /** @class */ (function (_super) {
    tslib_1.__extends(RangeSlider, _super);
    function RangeSlider(attrs) {
        var _this = _super.call(this, attrs) || this;
        _this.behaviour = "drag";
        _this.connected = [false, true, false];
        return _this;
    }
    RangeSlider.initClass = function () {
        this.prototype.default_view = RangeSliderView;
        this.override({
            format: "0[.]00",
        });
    };
    RangeSlider.prototype._formatter = function (value, format) {
        return numbro.format(value, format);
    };
    RangeSlider.__name__ = "RangeSlider";
    return RangeSlider;
}(abstract_slider_1.AbstractSlider));
exports.RangeSlider = RangeSlider;
RangeSlider.initClass();
