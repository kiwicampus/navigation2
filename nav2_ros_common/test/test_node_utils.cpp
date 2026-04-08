// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>

#include "nav2_ros_common/node_utils.hpp"
#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/copy_all_parameter_values.hpp"

using nav2::sanitize_node_name;
using nav2::generate_internal_node_name;
using nav2::generate_internal_node;
using nav2::add_namespaces;
using nav2::time_to_string;
using nav2::declare_parameter_if_not_declared;
using nav2::declare_or_get_parameter;
using nav2::get_plugin_type_param;

TEST(SanitizeNodeName, SanitizeNodeName)
{
  ASSERT_EQ(sanitize_node_name("bar"), "bar");
  ASSERT_EQ(sanitize_node_name("/foo/bar"), "_foo_bar");
}

TEST(TimeToString, IsLengthCorrect)
{
  ASSERT_EQ(time_to_string(0).length(), 0u);
  ASSERT_EQ(time_to_string(1).length(), 1u);
  ASSERT_EQ(time_to_string(10).length(), 10u);
  ASSERT_EQ(time_to_string(20)[0], '0');
}

TEST(TimeToString, TimeToStringDifferent)
{
  auto time1 = time_to_string(8);
  auto time2 = time_to_string(8);
  ASSERT_NE(time1, time2);
}

TEST(GenerateInternalNodeName, GenerateNodeName)
{
  auto defaultName = generate_internal_node_name();
  ASSERT_EQ(defaultName[0], '_');
  ASSERT_EQ(defaultName.length(), 9u);
}

TEST(AddNamespaces, AddNamespaceSlash)
{
  ASSERT_EQ(add_namespaces("hi", "bye"), "hi/bye");
  ASSERT_EQ(add_namespaces("hi/", "bye"), "/hi/bye");
}

TEST(DeclareParameterIfNotDeclared, DeclareParameterIfNotDeclared)
{
  auto node = std::make_shared<rclcpp::Node>("test_node");
  std::string param;

  // test declared parameter
  node->declare_parameter("foobar", "foo");
  declare_parameter_if_not_declared(node, "foobar", rclcpp::ParameterValue{"bar"});
  node->get_parameter("foobar", param);
  ASSERT_EQ(param, "foo");

  // test undeclared parameter
  declare_parameter_if_not_declared(node, "waldo", rclcpp::ParameterValue{"fred"});
  node->get_parameter("waldo", param);
  ASSERT_EQ(param, "fred");
}

TEST(DeclareOrGetParam, DeclareOrGetParam)
{
  auto node = std::make_shared<rclcpp::Node>("test_node");

  // test declared parameter
  node->declare_parameter("foobar", "foo");
  std::string param = declare_or_get_parameter(node, "foobar", std::string{"bar"});
  EXPECT_EQ(param, "foo");
  node->get_parameter("foobar", param);
  EXPECT_EQ(param, "foo");

  // test undeclared parameter
  node->set_parameter(rclcpp::Parameter("warn_on_missing_params", true));
  int int_param = declare_or_get_parameter(node, "waldo", 3);
  EXPECT_EQ(int_param, 3);

  // test unknown parameter with strict_param_loading enabled
  bool got_exception{false};
  node->set_parameter(rclcpp::Parameter("strict_param_loading", true));
  try {
    declare_or_get_parameter(node, "burpy", true);
  } catch (const rclcpp::exceptions::InvalidParameterValueException & exc) {
    got_exception = true;
  }
  EXPECT_TRUE(got_exception);
  // The parameter is anyway declared with the default val and subsequent calls won't fail
  EXPECT_TRUE(declare_or_get_parameter(node, "burpy", true));

  // test declaration by type of existing param
  int_param = declare_or_get_parameter<int>(node, "waldo");
  EXPECT_EQ(int_param, 3);

  // test declaration by type of non existing param
  got_exception = false;
  try {
    int_param = declare_or_get_parameter<int>(node, "wololo");
  } catch (const rclcpp::exceptions::InvalidParameterValueException & exc) {
    got_exception = true;
  }
  EXPECT_TRUE(got_exception);
}

TEST(GetPluginTypeParam, GetPluginTypeParam)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  auto node = std::make_shared<rclcpp::Node>("test_node");
  node->declare_parameter("Foo.plugin", "bar");
  ASSERT_EQ(get_plugin_type_param(node, "Foo"), "bar");
  EXPECT_THROW(get_plugin_type_param(node, "Waldo"), std::runtime_error);
}

TEST(TestParamCopying, TestParamCopying)
{
  auto node1 = std::make_shared<rclcpp::Node>("test_node1");
  auto node2 = std::make_shared<rclcpp::Node>("test_node2");

  // Tests for (1) multiple types, (2) recursion, (3) overriding values
  node1->declare_parameter("Foo1", rclcpp::ParameterValue(std::string("bar1")));
  node1->declare_parameter("Foo2", rclcpp::ParameterValue(0.123));
  node1->declare_parameter("Foo", rclcpp::ParameterValue(std::string("bar")));
  node1->declare_parameter("Foo.bar", rclcpp::ParameterValue(std::string("steve")));
  node2->declare_parameter("Foo", rclcpp::ParameterValue(std::string("barz2")));

  // Show Node2 is empty of Node1's parameters, but contains its own
  EXPECT_FALSE(node2->has_parameter("Foo1"));
  EXPECT_FALSE(node2->has_parameter("Foo2"));
  EXPECT_FALSE(node2->has_parameter("Foo.bar"));
  EXPECT_TRUE(node2->has_parameter("Foo"));
  EXPECT_EQ(node2->get_parameter("Foo").as_string(), std::string("barz2"));

  rclcpp::copy_all_parameter_values(node1, node2);

  // Test new parameters exist, of expected value, and original param is not overridden
  EXPECT_TRUE(node2->has_parameter("Foo1"));
  EXPECT_EQ(node2->get_parameter("Foo1").as_string(), std::string("bar1"));
  EXPECT_TRUE(node2->has_parameter("Foo2"));
  EXPECT_EQ(node2->get_parameter("Foo2").as_double(), 0.123);
  EXPECT_TRUE(node2->has_parameter("Foo.bar"));
  EXPECT_EQ(node2->get_parameter("Foo.bar").as_string(), std::string("steve"));
  EXPECT_TRUE(node2->has_parameter("Foo"));
  EXPECT_EQ(node2->get_parameter("Foo").as_string(), std::string("barz2"));
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  rclcpp::init(0, nullptr);

  int result = RUN_ALL_TESTS();

  rclcpp::shutdown();

  return result;
}
