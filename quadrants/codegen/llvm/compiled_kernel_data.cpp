#include "quadrants/codegen/llvm/compiled_kernel_data.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

namespace quadrants::lang {

namespace {

// CompiledKernelDataFile predates LLVM bitcode storage and has no independent
// payload-format field. Prefix new payloads so current builds can load bitcode
// without probing/parsing twice while old textual-IR artifacts remain readable.
constexpr char kBitcodeCachePrefix[] = "QDBC1";

}  // namespace

static std::unique_ptr<CompiledKernelData> new_llvm_compiled_kernel_data() {
  return std::make_unique<LLVM::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::llvm_creator = new_llvm_compiled_kernel_data;

namespace LLVM {

CompiledKernelData::CompiledKernelData(Arch arch, InternalData data) : arch_(arch), data_(std::move(data)) {
}

Arch CompiledKernelData::arch() const {
  return arch_;
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  return std::make_unique<CompiledKernelData>(arch_, data_);
}

CompiledKernelData::Err CompiledKernelData::check() const {
  const auto &compiled_data = data_.compiled_data;
  const auto &tasks = compiled_data.tasks;
  if (llvm::verifyModule(*compiled_data.module, &llvm::errs())) {
    return Err::kCompiledKernelDataBroken;
  }
  for (const auto &t : tasks) {
    if (compiled_data.module->getFunction(t.name) == nullptr) {
      return Err::kCompiledKernelDataBroken;
    }
  }
  return Err::kNoError;
}

std::string CompiledKernelData::debug_dump_to_string() const {
  auto &data = this->get_internal_data().compiled_data;
  auto *module = data.module.get();
  std::string result;
  llvm::raw_string_ostream oss(result);
  module->print(oss, /*AAW=*/nullptr);
  return oss.str();
}

CompiledKernelData::Err CompiledKernelData::load_impl(const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_llvm(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()), data_, true);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }

  const llvm::StringRef source(file.src_code());
  const llvm::StringRef bitcode_prefix(kBitcodeCachePrefix);
  std::unique_ptr<llvm::Module> module;
  if (source.size() >= bitcode_prefix.size() && source.take_front(bitcode_prefix.size()) == bitcode_prefix) {
    auto ret = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(source.drop_front(bitcode_prefix.size()), "cached_kernel_bitcode"), llvm_ctx_);
    if (!ret) {
      QD_DEBUG("Fail to parse cached LLVM bitcode");
      llvm::consumeError(ret.takeError());
      return Err::kParseSrcCodeFailed;
    }
    module = std::move(ret.get());
  } else {
    // Backward compatibility for artifacts written before the bitcode format.
    llvm::SMDiagnostic err;
    module = llvm::parseAssemblyString(source, err, llvm_ctx_);
    if (!module) {
      QD_DEBUG("Fail to parse llvm::Module from string: {}", err.getMessage().str());
      return Err::kParseSrcCodeFailed;
    }
  }
  data_.compiled_data.module = std::move(module);
  llvm::Module *mod = data_.compiled_data.module.get();
  mod->setModuleIdentifier("kernel");
  return Err::kNoError;
}

CompiledKernelData::Err CompiledKernelData::dump_impl(CompiledKernelDataFile &file) const {
  file.set_arch(arch_);
  try {
    file.set_metadata(liong::json::print(liong::json::serialize(data_)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  std::string bitcode;
  llvm::raw_string_ostream oss(bitcode);
  llvm::WriteBitcodeToFile(*data_.compiled_data.module, oss);
  oss.flush();

  std::string payload(kBitcodeCachePrefix);
  payload.append(bitcode);
  file.set_src_code(std::move(payload));
  return Err::kNoError;
}

}  // namespace LLVM
}  // namespace quadrants::lang
