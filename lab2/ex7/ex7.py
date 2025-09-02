import json

with open("catalog.json", "r") as f:
    products = json.load(f)

'''
1. Function that takes in a dictionary of products and displays the catalog in
a tabular format with appropriate column headers. The function should take the
dictionary of products as a parameter.
'''
def display_catalog(products_catalog):
    header = (
        f"{'ID':<6} {'Name':<20} {'Category':<15} "
        f"{'Price':<10} {'Quantity':<10} {'Description'}"
    )
    print(header)
    print("-" * 80)

    for pid, details in products_catalog.items():
        row = (
            f"{pid:<6} {details['name']:<20} {details['category']:<15} "
            f"{details['price']:<10.2f} {details['quantity']:<10} "
            f"{details['description']}"
        )
        print(row)

'''
2. Function that takes in a keyword and searches for products that contain the
keyword in their name. The function should return a dictionary with the product
IDs as keys and the product details as values. The function should take the
keyword as a parameter.

HINT: List Comprehension
'''
def search_product(keyword):
    filtered = {
        pid: details
        for pid, details in products.items()
        if keyword.lower() in details['name'].lower()
    }
    
    if filtered:
        display_catalog(filtered)
    else:
        print(f"No products found with keyword: '{keyword}'")


'''
3. Function that takes in a category and filters the products based on the
category. The function should return a dictionary with the product IDs as keys
and the product details as values. The function should take the category as a
parameter.

HINT: List Comprehension
'''
def filter_category(category):
    filtered = {
        pid: details
        for pid, details in products.items()
        if category.lower() in details['category'].lower()
    }
    
    if filtered:
        display_catalog(filtered)
    else:
        print(f"No products found with category: '{category}'")


'''
4. Function that adds a product to the shopping cart. The function should update
the quantity of the product in the cart if the product is already in the cart.
The function should also check if the quantity requested is available in stock
and update the stock accordingly. The function should take the product ID and
the quantity as parameters.
'''
def add_to_cart(shopping_cart, product_id, quantity):
    for i, (pid, qty) in enumerate(shopping_cart):
        if pid == product_id:
            if products[pid]['quantity'] - int(quantity) >= 0:
                shopping_cart[i] = (pid, qty + int(quantity))
                products[pid]['quantity'] -= int(quantity)
                return
            else:
                print("Not enough quatity in catalog")
                return
    
    shopping_cart.append((product_id, int(quantity)))
    products[product_id]['quantity'] -= int(quantity)


'''
5. Function that removes a product from the shopping cart. The function should
update the quantity of the product in the stock. The function should take the
product ID as a parameter.
'''
def remove_from_cart(shopping_cart, product_id):
    for i, (pid, qty) in enumerate(shopping_cart):
        if pid == product_id:
            del shopping_cart[i]
            print(f"Product '{pid}' removed from cart.")
            products[pid]['quantity'] += int(qty)
            return

'''
6. Function that calculates the total price of the products in the shopping cart.
The function should return the total price.
'''
def total(shopping_cart):
    total_price = 0
    for pid, qty in shopping_cart:
        total_price += products[pid]['price'] * qty

    print(f"\n*** Total: {total_price:.2f} ***")


'''
7. Function that displays the receipt of the purchase. The function should
display the products in the cart with their quantities and total price. The
function should also clear the shopping cart after displaying the receipt.
'''
def finalize_purchase(shopping_cart):

    if not shopping_cart:
        print(" *** Shopping cart is empty. ***")
        return
    else:                
        print("\n*** Receipt ***")
        print(f"{'ID':<6} {'Name':<20} {'Quantity':<10} {'Unit Price':<12} {'Total':<10}")
        print("-" * 80)

        total_price = 0
        total_articles = 0
        for pid, qty in shopping_cart:
            name = products[pid]['name']
            price = products[pid]['price']
            subtotal = price * qty
            total_price += subtotal
            total_articles += qty
            print(f"{pid:<6} {name:<20} {qty:<10} {price:<12.2f} {subtotal:<10.2f}")

        print("-" * 80)
        print(f"{'TOTAL ARTICLES':>60} {total_articles}")
        print(f"{'TOTAL PRICE':>60} {total_price:.2f}")
        shopping_cart.clear()
        print("Purchase finalized. Shopping cart cleared.")


if __name__ == "__main__":

    shopping_cart = []
    # TODO: loop for the shopping cart application

    while True:
        print("\nOptions:")
        print("1 - Display catalog")
        print("2 - Display cart")
        print("3 - Add to cart")
        print("4 - Remove from cart")
        print("5 - Search product")
        print("6 - Filter by category")
        print("7 - Calculate total")
        print("8 - Finalize purchase")
        print("9 - Exit")

        option = input("Select an option: ")
        if option == "1":
            display_catalog(products)
    
        elif option == "2":
            if not shopping_cart:
                print(" *** Shopping cart is empty. ***")
            else:
                total_articles = 0
                print(f"{'ID':<6} {'Name':<20} {'Quantity':<10} {'Unit Price':<12}")
                print("-" * 80)
                for pid, qty in shopping_cart:
                    name = products[pid]['name']
                    price = products[pid]['price']
                    total_articles += qty
                    print(f"{pid:<6} {name:<20} {qty:<10} {price:<12.2f}")
                print(f"Total number of articles: {total_articles}")

        elif option == "3":
            product_id = input("Enter product ID to add: ")
            quantity = input("Enter quantity: ")
            add_to_cart(shopping_cart, product_id, quantity)
        
        elif option == "4":
            product_id = input("Enter product ID to remove: ")
            found = any(pid == product_id for pid, _ in shopping_cart)

            if found:
                remove_from_cart(shopping_cart, product_id)
            else:
                print(f"Product '{product_id}' is not in the shopping cart")

        elif option == "5":
            keyword = input("Enter keyword: ")
            search_product(keyword)

        elif option == "6":
            category = input("Enter category to filter: ")
            filter_category(category)

        elif option == "7":
            total(shopping_cart)

        elif option == "8":
            finalize_purchase(shopping_cart)

        else:
            break
    
