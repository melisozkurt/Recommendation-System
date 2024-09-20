function showToast(header, content, style) {
  var toastEl = document.createElement('div');  // Create a new div element for the toast.
  toastEl.classList.add('toast'); // Add the "toast" class to the div.
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');
  toastEl.style.backgroundColor = "#fab007!important"
  // Set the toast style
  if (style) {
    toastEl.classList.add('bg-' + style);
  }

  // Create the toast header element
  var toastHeaderEl = document.createElement('div'); // Create a new div element for the toast header.
  toastHeaderEl.classList.add('toast-header');  // Add the "toast-header" class to the div.
  toastHeaderEl.style.backgroundColor = '#fff'
  toastHeaderEl.style.borderBottom = "none"
  toastHeaderEl.innerHTML = '<strong class="me-auto">' + header + '</strong>' +
    '<button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>';

  // Create the toast body element
  var toastBodyEl = document.createElement('div'); // Create a new div element for the toast body.
  toastBodyEl.classList.add('toast-body'); // Add the "toast-body" class to the div.
  toastBodyEl.textContent = content;
  if(style==="success"){
      toastBodyEl.style.backgroundColor = '#fab007'
  }

  // Add the header and body elements to the toast element
  toastEl.appendChild(toastHeaderEl);
  toastEl.appendChild(toastBodyEl);

  // Add the toast element to the container
  var toastContainerEl = document.querySelector('.toast-container');
  toastContainerEl.appendChild(toastEl);

  // Show the toast
  var toast = new bootstrap.Toast(toastEl); // Create a new bootstrap toast with the div.
  toast.show(); // Show the toast.
}

function onSubmitShowProducts() {


    if (!document.getElementById('products-main').querySelector('h3')) {
      const productsTitle = document.createElement("h3");
      productsTitle.innerText = 'PRODUCTS';
      document.getElementById('products-main').appendChild(productsTitle);
    }


    const userIndex = document.getElementById('userIndex').value
    console.log(userIndex)
    fetch("/show-products", {
        method: "POST",
        body: JSON.stringify({ userIndex: userIndex}),
        }).then((res) => {
        // Checks if the fetch request was successful.
        // If successful, display a success toast, otherwise, display an error toast and reload the page.
        if(res.ok){
        showToast('Success', 'The operation was successful!', 'success')
        res.json().then((data) => {

            document.getElementById('products').textContent = ''

            data.data.forEach(element => {
                const imageDiv = document.createElement("div");
                imageDiv.classList.add('product-image');

                const newImage = document.createElement("img");
                newImage.src = element.url

                const newTitle = document.createElement("h5");
                newTitle.innerText = element.title;

                const newRating = document.createElement("h5"); // Rating'i paragraf öğesi olarak oluşturun
                newRating.textContent = 'Rating: ' + element.rating; // Rating metnini ekleyin
            
                imageDiv.appendChild(newImage);
                imageDiv.appendChild(newTitle);
                imageDiv.appendChild(newRating)


                document.getElementById('products').appendChild(imageDiv)
            })
            }
        )
        }else{
        showToast('Error', res.statusText, 'danger');
        }
  });
}

function onSubmitShowSuggestions() {

  if (!document.getElementById('suggestions-main').querySelector('h3')) {
    const sugTitle = document.createElement("h3");
    sugTitle.innerText = 'SUGGESTIONS';
    document.getElementById('suggestions-main').appendChild(sugTitle);
  }
  document.getElementById('suggestions').textContent = ''


  const userIndex = document.getElementById('userIndex').value;
  const noproducts = document.getElementById('noProducts').value;
  const selectedAlgorithm = document.getElementById('algorithm').value;

  fetch("/show-suggestions", {
      method: "POST",
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({
          userIndex: userIndex,
          noproducts: noproducts,
          selectedAlgorithm: selectedAlgorithm
      }),
  })
  .then((res) => {
      if (res.ok) {
          showToast('Success', 'The operation was successful!', 'success');
          return res.json();
      } else {
          showToast('Error', res.statusText, 'danger');
          throw new Error(res.statusText);
      }
  })
  .then((data) => {

      document.getElementById('suggestions').textContent = ''

      data.data.forEach(element => {
          const imageDiv = document.createElement("div");
          imageDiv.classList.add('suggestions-image');

          const newImage = document.createElement("img");
          newImage.src = element.url;

          const newTitle = document.createElement("h5");
          newTitle.innerText = element.title;

          imageDiv.appendChild(newImage);
          imageDiv.appendChild(newTitle);

          document.getElementById('suggestions').appendChild(imageDiv)
      });
  })
  .catch((error) => {
      showToast('Error', error.message, 'danger');
  });
}