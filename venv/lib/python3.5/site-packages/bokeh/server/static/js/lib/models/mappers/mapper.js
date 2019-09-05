"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var transform_1 = require("../transforms/transform");
var Mapper = /** @class */ (function (_super) {
    tslib_1.__extends(Mapper, _super);
    function Mapper(attrs) {
        return _super.call(this, attrs) || this;
    }
    Mapper.prototype.compute = function (_x) {
        // If it's just a single value, then a mapper doesn't really make sense.
        throw new Error("mapping single values is not supported");
    };
    Mapper.__name__ = "Mapper";
    return Mapper;
}(transform_1.Transform));
exports.Mapper = Mapper;
