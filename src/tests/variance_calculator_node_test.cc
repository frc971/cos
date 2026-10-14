#include "localization/variance_calculator_node.h"

#include <chrono>
#include <memory>
#include <stop_token>

#include <gtest/gtest.h>

#include "control_loop/context.h"
#include "localization/position.h"

namespace {

auto MakeContext() -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
}

void ExpectNullOutputAndCallback(
    localization::VarianceCalculatorNode& node,
    const control_loop::Context& context) {
  int callback_count = 0;
  node.RegisterCallback(
      [&callback_count](const control_loop::Context& callback_context) {
        ++callback_count;
        EXPECT_TRUE(callback_context->Exists("output"));
        EXPECT_EQ(callback_context
                      ->GetMessage<localization::PositionEstimateMessage>(
                          "output"),
                  nullptr);
      });

  node.CreateCallback()(context);

  EXPECT_EQ(callback_count, 1);
}

TEST(VarianceCalculatorNodeTest, NotifiesCallbackForMissingInput) {
  localization::VarianceCalculatorNode node("input", "output");

  ExpectNullOutputAndCallback(node, MakeContext());
}

TEST(VarianceCalculatorNodeTest, NotifiesCallbackForEmptyDistances) {
  localization::VarianceCalculatorNode node("input", "output");
  auto context = MakeContext();
  auto input = std::make_unique<localization::PositionEstimateMessage>();
  input->tag_ids = {1};
  context->SetMessage("input", std::move(input));

  ExpectNullOutputAndCallback(node, context);
}

TEST(VarianceCalculatorNodeTest, NotifiesCallbackForMismatchedTagCount) {
  localization::VarianceCalculatorNode node("input", "output");
  auto context = MakeContext();
  auto input = std::make_unique<localization::PositionEstimateMessage>();
  input->distances = {2.0};
  context->SetMessage("input", std::move(input));

  ExpectNullOutputAndCallback(node, context);
}

TEST(VarianceCalculatorNodeTest, PublishesVarianceBeforeNotifyingCallback) {
  localization::VarianceCalculatorNode node("input", "output");
  auto context = MakeContext();
  auto input = std::make_unique<localization::PositionEstimateMessage>();
  input->tag_ids = {1, 2};
  input->distances = {2.0, 4.0};
  input->timestamp = 12.5;
  context->SetMessage("input", std::move(input));

  int callback_count = 0;
  node.RegisterCallback(
      [&callback_count](const control_loop::Context& callback_context) {
        ++callback_count;
        const auto* output =
            callback_context
                ->GetMessage<localization::PositionEstimateMessage>("output");
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(output->tag_ids, (std::vector<int>{1, 2}));
        EXPECT_EQ(output->distances, (std::vector<double>{2.0, 4.0}));
        EXPECT_DOUBLE_EQ(output->variance, 1.525);
        EXPECT_DOUBLE_EQ(output->timestamp, 12.5);
      });

  node.CreateCallback()(context);

  EXPECT_EQ(callback_count, 1);
}

}  // namespace
