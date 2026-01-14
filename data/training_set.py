"""Training examples for DSPy NL-to-SQL optimization."""
import dspy


def get_training_examples(schema_info_str: str) -> list:
    """
    Returns ~35 dspy.Examples for training the SQL generator.
    
    Coverage areas:
    1. Date handling (1997 -> 2017 shift, BETWEEN syntax)
    2. Table aliases (cat=Categories, c=customers, o=orders, od=order_items, p=products)
    3. KPI formulas (Revenue, AOV, Margin)
    4. Complex JOINs
    5. Edge cases and failure modes
    """
    
    ctx_summer = "Marketing Calendar: Summer Beverages 1997 = 2017-06-01 to 2017-06-30."
    ctx_winter = "Marketing Calendar: Winter Classics 1997 = 2017-12-01 to 2017-12-31."
    ctx_kpi = "Revenue = SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)). AOV = Revenue / COUNT(DISTINCT o.OrderID)."
    ctx_margin = "Gross Margin = Revenue - Cost. Cost = 0.7 * UnitPrice when unknown."
    
    examples = [
        # GROUP A: DATE HANDLING
        dspy.Example(
            question="How many orders were placed in 1997?",
            schema_info=schema_info_str,
            context="DB dates are shifted +20 years (1997->2017).",
            sql_query="SELECT COUNT(*) FROM orders o WHERE o.OrderDate BETWEEN '2017-01-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Total revenue for Summer Beverages 1997",
            schema_info=schema_info_str,
            context=ctx_summer + " " + ctx_kpi,
            sql_query="SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Beverages' AND o.OrderDate BETWEEN '2017-06-01' AND '2017-06-30'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="What was the average order value in December 1997?",
            schema_info=schema_info_str,
            context=ctx_winter + " " + ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) FROM orders o JOIN order_items od ON o.OrderID = od.OrderID WHERE o.OrderDate BETWEEN '2017-12-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Show orders between Jan 1st and Jan 31st 1997",
            schema_info=schema_info_str,
            context="DB dates shifted +20 years.",
            sql_query="SELECT * FROM orders o WHERE o.OrderDate BETWEEN '2017-01-01' AND '2017-01-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Orders placed between June and August 1997",
            schema_info=schema_info_str,
            context="DB dates shifted +20 years.",
            sql_query="SELECT o.OrderID, o.OrderDate FROM orders o WHERE o.OrderDate BETWEEN '2017-06-01' AND '2017-08-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        # GROUP B: ALIAS USAGE (cat for Categories, c for customers)
        dspy.Example(
            question="List all products in the Seafood category",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT p.ProductName FROM products p JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Seafood'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Show total quantity sold for Dairy Products",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT SUM(od.Quantity) FROM order_items od JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Dairy Products'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Products in Condiments category with price > 20",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT p.ProductName, p.UnitPrice FROM products p JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Condiments' AND p.UnitPrice > 20"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Who are the customers from Germany?",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT c.CompanyName, c.ContactName FROM customers c WHERE c.Country = 'Germany'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Find the customer with the most orders",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT c.CompanyName, COUNT(o.OrderID) AS OrderCount FROM orders o JOIN customers c ON o.CustomerID = c.CustomerID GROUP BY c.CompanyName ORDER BY OrderCount DESC LIMIT 1"
        ).with_inputs("question", "schema_info", "context"),
        
        # GROUP C: KPI FORMULAS
        dspy.Example(
            question="Top 3 products by total revenue all-time",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue FROM order_items od JOIN products p ON od.ProductID = p.ProductID GROUP BY p.ProductName ORDER BY Revenue DESC LIMIT 3"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="What is the gross margin for customer 'Vins et alcools Chevalier'?",
            schema_info=schema_info_str,
            context=ctx_margin,
            sql_query="SELECT c.CompanyName, SUM((od.UnitPrice * od.Quantity * (1 - od.Discount)) - (0.7 * od.UnitPrice * od.Quantity * (1 - od.Discount))) AS Margin FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN customers c ON o.CustomerID = c.CustomerID WHERE c.CompanyName = 'Vins et alcools Chevalier' GROUP BY c.CompanyName"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Top 5 employees by sales amount",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT e.FirstName, e.LastName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Sales FROM Employees e JOIN orders o ON e.EmployeeID = o.EmployeeID JOIN order_items od ON o.OrderID = od.OrderID GROUP BY e.EmployeeID ORDER BY Sales DESC LIMIT 5"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Revenue by category for all time",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT cat.CategoryName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue FROM order_items od JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID GROUP BY cat.CategoryName ORDER BY Revenue DESC"
        ).with_inputs("question", "schema_info", "context"),
        
        # GROUP D: BASIC QUERIES
        dspy.Example(
            question="List top 10 most expensive products",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT ProductName, UnitPrice FROM products ORDER BY UnitPrice DESC LIMIT 10"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="How many different products do we sell?",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT COUNT(*) FROM products"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="What is the total discount given on all orders?",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT SUM(od.UnitPrice * od.Quantity * od.Discount) FROM order_items od"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="List products with zero stock",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT ProductName FROM products WHERE UnitsInStock = 0"
        ).with_inputs("question", "schema_info", "context"),
        
        # GROUP E: FAILURE MODE TARGETING - AOV
        dspy.Example(
            question="What was the Average Order Value during Winter Classics 1997?",
            schema_info=schema_info_str,
            context=ctx_winter + " " + ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) AS AOV FROM orders o JOIN order_items od ON o.OrderID = od.OrderID WHERE o.OrderDate BETWEEN '2017-12-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Calculate AOV for Summer Beverages 1997",
            schema_info=schema_info_str,
            context=ctx_summer + " " + ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) AS AOV FROM orders o JOIN order_items od ON o.OrderID = od.OrderID WHERE o.OrderDate BETWEEN '2017-06-01' AND '2017-06-30'"
        ).with_inputs("question", "schema_info", "context"),
        
        # GROUP F: FAILURE MODE TARGETING - MARGIN WITH 0.7
        dspy.Example(
            question="Who was the top customer by gross margin in 1997?",
            schema_info=schema_info_str,
            context=ctx_margin + " DB dates +20 years.",
            sql_query="SELECT c.CompanyName, SUM((od.UnitPrice * od.Quantity * (1 - od.Discount)) - (0.7 * od.UnitPrice * od.Quantity * (1 - od.Discount))) AS margin FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN customers c ON o.CustomerID = c.CustomerID WHERE o.OrderDate BETWEEN '2017-01-01' AND '2017-12-31' GROUP BY c.CompanyName ORDER BY margin DESC LIMIT 1"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Gross margin for Beverages category",
            schema_info=schema_info_str,
            context=ctx_margin,
            sql_query="SELECT SUM((od.UnitPrice * od.Quantity * (1 - od.Discount)) - (0.7 * od.UnitPrice * od.Quantity * (1 - od.Discount))) AS margin FROM order_items od JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Beverages'"
        ).with_inputs("question", "schema_info", "context"),
        
        # GROUP G: COMPLEX JOINS WITH CORRECT ALIASES
        dspy.Example(
            question="Which customers ordered Beverages in 1997?",
            schema_info=schema_info_str,
            context="DB dates +20 years.",
            sql_query="SELECT DISTINCT c.CompanyName FROM orders o JOIN customers c ON o.CustomerID = c.CustomerID JOIN order_items od ON o.OrderID = od.OrderID JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Beverages' AND o.OrderDate BETWEEN '2017-01-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Categories with more than 10 products",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT cat.CategoryName, COUNT(p.ProductID) AS ProductCount FROM products p JOIN Categories cat ON p.CategoryID = cat.CategoryID GROUP BY cat.CategoryName HAVING COUNT(p.ProductID) > 10"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Top customer by order count in December 1997",
            schema_info=schema_info_str,
            context=ctx_winter,
            sql_query="SELECT c.CompanyName, COUNT(o.OrderID) AS OrderCount FROM orders o JOIN customers c ON o.CustomerID = c.CustomerID WHERE o.OrderDate BETWEEN '2017-12-01' AND '2017-12-31' GROUP BY c.CompanyName ORDER BY OrderCount DESC LIMIT 1"
        ).with_inputs("question", "schema_info", "context"),
        
        # NEW EXAMPLES (11 more targeting failure modes)
        
        # 1. ROUND with float output
        dspy.Example(
            question="Total revenue from Beverages during Summer 1997 rounded to 2 decimals",
            schema_info=schema_info_str,
            context=ctx_summer + " " + ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS Revenue FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Beverages' AND o.OrderDate BETWEEN '2017-06-01' AND '2017-06-30'"
        ).with_inputs("question", "schema_info", "context"),
        
        # 2. Category with date filter
        dspy.Example(
            question="Which category had highest quantity sold during Summer 1997?",
            schema_info=schema_info_str,
            context=ctx_summer,
            sql_query="SELECT cat.CategoryName, SUM(od.Quantity) AS total_quantity FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE o.OrderDate BETWEEN '2017-06-01' AND '2017-06-30' GROUP BY cat.CategoryName ORDER BY total_quantity DESC LIMIT 1"
        ).with_inputs("question", "schema_info", "context"),
        
        # 3. Simple count with date
        dspy.Example(
            question="How many orders in December 1997?",
            schema_info=schema_info_str,
            context=ctx_winter,
            sql_query="SELECT COUNT(*) FROM orders o WHERE o.OrderDate BETWEEN '2017-12-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        # 4. Revenue without date
        dspy.Example(
            question="Total revenue from all Condiments",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue FROM order_items od JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Condiments'"
        ).with_inputs("question", "schema_info", "context"),
        
        # 5. Customer with margin calculation
        dspy.Example(
            question="Top 3 customers by gross margin",
            schema_info=schema_info_str,
            context=ctx_margin,
            sql_query="SELECT c.CompanyName, SUM((od.UnitPrice * od.Quantity * (1 - od.Discount)) - (0.7 * od.UnitPrice * od.Quantity * (1 - od.Discount))) AS margin FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN customers c ON o.CustomerID = c.CustomerID GROUP BY c.CompanyName ORDER BY margin DESC LIMIT 3"
        ).with_inputs("question", "schema_info", "context"),
        
        # 6. AOV without date filter
        dspy.Example(
            question="What is the overall average order value?",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) AS AOV FROM orders o JOIN order_items od ON o.OrderID = od.OrderID"
        ).with_inputs("question", "schema_info", "context"),
        
        # 7. Product count by category
        dspy.Example(
            question="How many products in each category?",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT cat.CategoryName, COUNT(p.ProductID) AS ProductCount FROM products p JOIN Categories cat ON p.CategoryID = cat.CategoryID GROUP BY cat.CategoryName"
        ).with_inputs("question", "schema_info", "context"),
        
        # 8. Customer orders with amount
        dspy.Example(
            question="Total order amount for customer 'Ernst Handel'",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT c.CompanyName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalAmount FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN customers c ON o.CustomerID = c.CustomerID WHERE c.CompanyName = 'Ernst Handel' GROUP BY c.CompanyName"
        ).with_inputs("question", "schema_info", "context"),
        
        # 9. Products ordered by a country
        dspy.Example(
            question="Products ordered by customers from France",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT DISTINCT p.ProductName FROM orders o JOIN customers c ON o.CustomerID = c.CustomerID JOIN order_items od ON o.OrderID = od.OrderID JOIN products p ON od.ProductID = p.ProductID WHERE c.Country = 'France'"
        ).with_inputs("question", "schema_info", "context"),
        
        # 10. Revenue comparison
        dspy.Example(
            question="Revenue from Seafood vs Meat/Poultry",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT cat.CategoryName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue FROM order_items od JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName IN ('Seafood', 'Meat/Poultry') GROUP BY cat.CategoryName"
        ).with_inputs("question", "schema_info", "context"),
        
        # 11. Winter campaign revenue
        dspy.Example(
            question="Total revenue during Winter Classics 1997",
            schema_info=schema_info_str,
            context=ctx_winter + " " + ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS Revenue FROM orders o JOIN order_items od ON o.OrderID = od.OrderID WHERE o.OrderDate BETWEEN '2017-12-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
    ]
    
    return examples


def get_validation_examples(schema_info_str: str) -> list:
    """Returns examples for validation during optimization."""
    ctx_kpi = "Revenue = SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)). AOV = Revenue / COUNT(DISTINCT o.OrderID)."
    ctx_margin = "Gross Margin = Revenue - Cost. Cost = 0.7 * UnitPrice when unknown."
    
    return [
        dspy.Example(
            question="Total orders in 1997",
            schema_info=schema_info_str,
            context="DB dates +20 years.",
            sql_query="SELECT COUNT(*) FROM orders o WHERE o.OrderDate BETWEEN '2017-01-01' AND '2017-12-31'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Revenue from Seafood category",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) FROM order_items od JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID WHERE cat.CategoryName = 'Seafood'"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Top customer by total orders",
            schema_info=schema_info_str,
            context="",
            sql_query="SELECT c.CompanyName, COUNT(o.OrderID) AS OrderCount FROM orders o JOIN customers c ON o.CustomerID = c.CustomerID GROUP BY c.CompanyName ORDER BY OrderCount DESC LIMIT 1"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="AOV for all orders",
            schema_info=schema_info_str,
            context=ctx_kpi,
            sql_query="SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) FROM orders o JOIN order_items od ON o.OrderID = od.OrderID"
        ).with_inputs("question", "schema_info", "context"),
        
        dspy.Example(
            question="Top customer by gross margin",
            schema_info=schema_info_str,
            context=ctx_margin,
            sql_query="SELECT c.CompanyName, SUM((od.UnitPrice * od.Quantity * (1 - od.Discount)) - (0.7 * od.UnitPrice * od.Quantity * (1 - od.Discount))) AS margin FROM orders o JOIN order_items od ON o.OrderID = od.OrderID JOIN customers c ON o.CustomerID = c.CustomerID GROUP BY c.CompanyName ORDER BY margin DESC LIMIT 1"
        ).with_inputs("question", "schema_info", "context"),
    ]
